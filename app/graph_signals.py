"""Knowledge-graph signals for the re-ranking stage (fase B3, memoria
Sección sec:met-reranking).

For each candidate researcher the graph contributes evidence the aggregated
biography does not express directly:

  - supervision experience: how many TFG/TFM/theses they have supervised;
  - thematic fit of what they supervised: keywords/titles of their
    LastYearProject nodes vs the query;
  - teaching: subjects taught;
  - co-authorship community: how connected the candidate is WITHIN the
    retrieved candidate set (shared Publication nodes).

Two consumers (the two integration mechanisms compared in B4):
  1. `evidence_lines()` — compact, readable lines injected into the listwise
     prompt so the LLM reasons over the evidence (features-in-prompt, the main
     proposal). Content is passed raw (Spanish/English mix); the multilingual
     re-ranking model judges cross-lingual fit itself.
  2. `components()` — per-candidate NUMERIC signals stored in
     `rerank_components` so the ranking can be re-fused off-line under many
     weights/subsets (late fusion + sensitivity analysis) without re-invoking
     the LLM. The lexical `*_fit` signals use diacritic-insensitive containment
     and are limited by the language gap (English query vs mostly-Spanish
     titles/subjects) — a documented limitation; the prompt mechanism does not
     share it.

Everything is loaded from Memgraph ONCE per evaluation run (batched over the
whole candidate directory, ~750 PDIs) and kept in memory: per-query work is
pure Python, so the re-ranking latency stays LLM-dominated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from utils.text import normalize_topic

# Caps keeping prompt lines compact and signal computation bounded.
_RECENT_WORKS = 20  # most recent supervised works considered per PDI
_PROMPT_TOPICS = 8
_PROMPT_SUBJECTS = 5
_MIN_TERM_LEN = 4  # ignore ultra-short terms in lexical fit (noise like "ia")


@dataclass
class PdiGraphInfo:
    """Static (query-independent) graph facts for one researcher."""

    supervised_total: int = 0
    supervised_types: dict[str, int] = field(default_factory=dict)
    supervised_topics: list[str] = field(default_factory=list)  # recent, deduped
    supervised_titles: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)


class SignalStore:
    """In-memory graph signals for a fixed candidate directory."""

    def __init__(
        self,
        info: dict[str, PdiGraphInfo],
        coauthors: dict[str, set[str]],
    ):
        self._info = info
        self._coauthors = coauthors

    # --- mechanism 1: readable evidence for the prompt -----------------------
    def evidence_lines(self, researcher_id: str) -> list[str]:
        """Compact evidence lines for one candidate (empty when no graph data)."""
        pdi = self._info.get(researcher_id)
        if pdi is None:
            return []
        lines: list[str] = []
        if pdi.supervised_total:
            types = ", ".join(
                f"{count} {name}" for name, count in sorted(pdi.supervised_types.items())
            )
            line = f"Supervised student works: {pdi.supervised_total} ({types})"
            topics = pdi.supervised_topics[:_PROMPT_TOPICS] or pdi.supervised_titles[:3]
            if topics:
                line += f"; recent supervised topics: {', '.join(topics)}"
            lines.append(line)
        if pdi.subjects:
            lines.append(f"Teaches: {', '.join(pdi.subjects[:_PROMPT_SUBJECTS])}")
        return lines

    # --- mechanism 2: numeric components for late fusion / sensitivity -------
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalise the query text once per query for the lexical fit signals.

        Callers should concatenate the ENRICHED English query AND the raw
        (Spanish) student proposal before normalising: supervised-work keywords
        and subject names are mostly Spanish, so matching against both texts
        lets each language find its counterpart (verified: an English-only query
        misses topics like «educación» that its Spanish original does match).
        """
        return normalize_topic(query or "")

    def components(
        self,
        researcher_id: str,
        normalized_query: str,
        candidate_set: set[str],
    ) -> dict[str, float]:
        """Numeric graph signals for one candidate against one query.

        Raw values (counts), NOT normalised or weighted: the off-line sweep
        decides transformations and weights.
        """
        pdi = self._info.get(researcher_id) or PdiGraphInfo()
        supervised_fit = self._lexical_fit(pdi.supervised_topics, normalized_query)
        teaching_fit = self._lexical_fit(pdi.subjects, normalized_query)
        degree = len(self._coauthors.get(researcher_id, set()) & candidate_set)
        return {
            "supervised_total": float(pdi.supervised_total),
            "supervised_fit": float(supervised_fit),
            "teaching_fit": float(teaching_fit),
            "coauthors_in_set": float(degree),
        }

    @staticmethod
    def _lexical_fit(terms: list[str], normalized_query: str) -> int:
        """Distinct terms contained in the normalised query (containment match)."""
        if not normalized_query:
            return 0
        matched = 0
        seen: set[str] = set()
        for term in terms:
            norm = normalize_topic(term)
            if len(norm) < _MIN_TERM_LEN or norm in seen:
                continue
            seen.add(norm)
            if norm in normalized_query:
                matched += 1
        return matched


def _dedup(values: list) -> list[str]:
    """Flatten/trim/dedupe keyword-ish values preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            text = str(item or "").strip()
            low = text.lower()
            if text and low not in seen:
                seen.add(low)
                out.append(text)
    return out


def load_signal_store(researcher_ids: list[str]) -> SignalStore | None:
    """Batched one-shot load of every signal for a candidate directory.

    Three UNWIND/IN queries over the given ids (supervision, teaching,
    co-authorship among the ids). Returns None when Memgraph is unavailable so
    the caller can skip the graph-grounded mode gracefully.
    """
    try:
        from dbs.graph.memgraph import MemgraphBackend  # lazy: needs MEMGRAPH_* env

        db = MemgraphBackend()

        info: dict[str, PdiGraphInfo] = {}
        for row in db.query(
            """
            UNWIND $ids AS rid
            MATCH (p:Pdi {id: rid})-[:Tutoring]->(t:LastYearProject)
            WITH rid, t ORDER BY t.year DESC
            RETURN rid,
                   count(t) AS total,
                   collect(t.type) AS types,
                   collect(t.keywords)[0..$recent] AS keywords,
                   collect(t.name)[0..$recent] AS titles
            """,
            {"ids": researcher_ids, "recent": _RECENT_WORKS},
        ):
            types: dict[str, int] = {}
            for t in row.get("types") or []:
                name = str(t or "").strip() or "otros"
                types[name] = types.get(name, 0) + 1
            info[row["rid"]] = PdiGraphInfo(
                supervised_total=int(row.get("total") or 0),
                supervised_types=types,
                supervised_topics=_dedup(row.get("keywords") or []),
                supervised_titles=_dedup(row.get("titles") or []),
            )

        for row in db.query(
            """
            UNWIND $ids AS rid
            MATCH (p:Pdi {id: rid})-[e:Teaching]->(s:Subject)
            WITH rid, s.name AS name, max(e.end_date) AS until
            ORDER BY until DESC
            RETURN rid, collect(name) AS subjects
            """,
            {"ids": researcher_ids},
        ):
            pdi = info.setdefault(row["rid"], PdiGraphInfo())
            pdi.subjects = _dedup(row.get("subjects") or [])

        coauthors: dict[str, set[str]] = {}
        for row in db.query(
            """
            MATCH (a:Pdi)-[:Authorship]->(:Publication)<-[:Authorship]-(b:Pdi)
            WHERE a.id IN $ids AND b.id IN $ids AND a.id < b.id
            RETURN a.id AS a, b.id AS b
            """,
            {"ids": researcher_ids},
        ):
            coauthors.setdefault(row["a"], set()).add(row["b"])
            coauthors.setdefault(row["b"], set()).add(row["a"])

        logger.info(
            "Graph signal store: {} PDIs with supervision/teaching data, "
            "{} with co-authors inside the directory ({} ids requested).",
            len(info),
            len(coauthors),
            len(researcher_ids),
        )
        return SignalStore(info, coauthors)
    except Exception:  # noqa: BLE001 — degrade: the kg mode is skipped upstream
        logger.exception("Memgraph unavailable — graph signals disabled")
        return None
