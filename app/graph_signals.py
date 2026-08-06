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
class SupervisedWork:
    """One supervised work as the signals see it, kept individually.

    Storing the works separately (instead of a pre-flattened topic list) is what
    makes an exact leave-one-out possible: masking a work must also free its slot
    in the recency window, and must not drop a topic that another work shares.
    """

    work_id: str = ""
    year: int = 0
    type: str = ""
    keywords: list[str] = field(default_factory=list)
    title: str = ""


@dataclass
class PdiGraphInfo:
    """Static (query-independent) graph facts for one researcher."""

    works: list[SupervisedWork] = field(default_factory=list)  # newest first
    subjects: list[str] = field(default_factory=list)

    def view(self, exclude_work_id: str | None = None) -> "SupervisionView":
        """Supervision facts as the signals consume them, optionally masking one work.

        `exclude_work_id` drops that work before applying the recency window, so
        the view is exactly what would have been computed had the work never
        existed. Used by the leave-one-out protocol of the evaluation, where the
        work that originates a query must not inform the signals of its own
        supervisor (memoria sec:eval-validez).
        """
        works = [w for w in self.works if w.work_id != exclude_work_id]
        recent = works[:_RECENT_WORKS]
        types: dict[str, int] = {}
        for w in works:
            name = (w.type or "").strip() or "otros"
            types[name] = types.get(name, 0) + 1
        return SupervisionView(
            total=len(works),
            types=types,
            topics=_dedup([kw for w in recent for kw in w.keywords]),
            titles=_dedup([w.title for w in recent]),
        )


@dataclass
class SupervisionView:
    """Supervision facts for one candidate, already windowed and deduplicated."""

    total: int = 0
    types: dict[str, int] = field(default_factory=dict)
    topics: list[str] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)


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
    def evidence_lines(
        self, researcher_id: str, exclude_work_id: str | None = None
    ) -> list[str]:
        """Compact evidence lines for one candidate (empty when no graph data).

        `exclude_work_id` hides one supervised work, so the re-ranker cannot read
        back the very work that produced the query (leave-one-out protocol).
        """
        pdi = self._info.get(researcher_id)
        if pdi is None:
            return []
        sup = pdi.view(exclude_work_id)
        lines: list[str] = []
        if sup.total:
            types = ", ".join(
                f"{count} {name}" for name, count in sorted(sup.types.items())
            )
            line = f"Supervised student works: {sup.total} ({types})"
            topics = sup.topics[:_PROMPT_TOPICS] or sup.titles[:3]
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
        exclude_work_id: str | None = None,
    ) -> dict[str, float]:
        """Numeric graph signals for one candidate against one query.

        Raw values (counts), NOT normalised or weighted: the off-line sweep
        decides transformations and weights.

        `exclude_work_id` masks one supervised work from the supervision signals
        (`supervised_total`, `supervised_fit`), which is how the leave-one-out
        protocol removes the echo of the query's own work. The teaching and
        co-authorship signals are unaffected: neither derives from the works
        supervised, so both are clean by construction.
        """
        pdi = self._info.get(researcher_id) or PdiGraphInfo()
        sup = pdi.view(exclude_work_id)
        supervised_fit = self._lexical_fit(sup.topics, normalized_query)
        teaching_fit = self._lexical_fit(pdi.subjects, normalized_query)
        degree = len(self._coauthors.get(researcher_id, set()) & candidate_set)
        return {
            "supervised_total": float(sup.total),
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
                   collect({id: t.id, year: t.year, type: t.type,
                            keywords: t.keywords, name: t.name}) AS works
            """,
            {"ids": researcher_ids},
        ):
            works = [
                SupervisedWork(
                    work_id=str(w.get("id") or ""),
                    year=int(w.get("year") or 0),
                    type=str(w.get("type") or ""),
                    keywords=[str(k) for k in (w.get("keywords") or [])],
                    title=str(w.get("name") or ""),
                )
                for w in (row.get("works") or [])
            ]
            info[row["rid"]] = PdiGraphInfo(works=works)

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
