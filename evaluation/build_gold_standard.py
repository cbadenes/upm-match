"""
Build the gold standard dataset for UPM Match evaluation.

Creates two JSON files:
  evaluation/gold_standard_tfg.json  — 250 TFG records
  evaluation/gold_standard_tfm.json  — 250 TFM records

Selection criteria:
  - PDIs are sampled proportionally to the total number of PDIs per school
    (e.g. if ETSIST holds ~15% of all PDIs, ~15% of the records come from
    ETSIST directors).
  - One work per selected PDI: the most interesting work that has an abstract.
  - If a school cannot supply enough PDIs, the shortfall is topped up from the
    remaining PDIs of other schools (ranked by interestingness) so the total
    still reaches 250.

Run: uv run evaluation/build_gold_standard.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from models import ResearcherEnrichmentDocument, ResearcherLastYearProject
from utils.researchers import load_researcher_ids, load_enrichment

# School affiliation lookup (position id → school abbreviation).
TARGET_SCHOOL_IDS = {
    "loc_16086": "ETSIT",
    "loc_16097": "ETSII",
    "loc_16331": "ETSISI",
    "loc_16380": "ETSIST",
}
DEPT_TO_SCHOOL = {
    "loc_16408": "ETSISI",
    "loc_16409": "ETSIST",
    "loc_16410": "ETSIST",
    "loc_16411": "ETSIST",
    "loc_16431": "ETSII",
    "loc_16432": "ETSII",
    "loc_16433": "ETSII",
    "loc_16437": "ETSIT",
    "loc_16438": "ETSIT",
    "loc_16439": "ETSIT",
    "loc_16440": "ETSIT",
    "loc_16441": "ETSIT",
    "loc_16442": "ETSIT",
}
ALL_LOC_TO_SCHOOL = {**TARGET_SCHOOL_IDS, **DEPT_TO_SCHOOL}

# Evaluation targets.
TARGET_TOTAL = 250         # records per study type (TFG / TFM)
MIN_ABSTRACT_LEN = 150     # a work needs a real abstract to be eligible


def school_of(doc: ResearcherEnrichmentDocument) -> str:
    for pos in doc.positions:
        if pos.id in ALL_LOC_TO_SCHOOL:
            return ALL_LOC_TO_SCHOOL[pos.id]
    return "UNKNOWN"


def interestingness(proj: ResearcherLastYearProject) -> int:
    """Score a work for selection priority. Richer abstract + keywords + recency."""
    score = 0
    abstract = (proj.abstract or "").strip()
    score += min(len(abstract) // 200, 6)   # 0-6 pts for abstract richness
    if proj.keywords:
        score += min(len(proj.keywords), 3)  # 0-3 pts for keywords
    if proj.year and proj.year >= 2020:
        score += 1                           # recency bonus
    return score


def _build_description(proj: ResearcherLastYearProject) -> str:
    """Combine title + abstract + keywords into a rich description string."""
    parts = []
    if proj.title:
        parts.append(proj.title)
    if proj.abstract:
        parts.append(proj.abstract.strip())
    if proj.keywords:
        parts.append("Keywords: " + ", ".join(proj.keywords))
    return "\n\n".join(parts)


def eligible_works(
    doc: ResearcherEnrichmentDocument, study_type: str
) -> list[ResearcherLastYearProject]:
    """Works of `study_type` that have a real abstract, best first."""
    candidates = [
        proj
        for proj in doc.last_year_projects
        if proj.type == study_type
        and proj.title
        and len((proj.abstract or "").strip()) >= MIN_ABSTRACT_LEN
    ]
    return sorted(candidates, key=interestingness, reverse=True)


def build_candidate_pool(
    docs: list[ResearcherEnrichmentDocument],
    study_type: str,
) -> dict[str, list[list[dict]]]:
    """school -> list of PDI groups (each group = that PDI's eligible works,
    best first). Groups are sorted by their best work's interestingness desc."""
    pool: dict[str, list[list[dict]]] = {
        school: [] for school in set(ALL_LOC_TO_SCHOOL.values())
    }
    for doc in docs:
        school = school_of(doc)
        if school == "UNKNOWN":
            continue
        works = eligible_works(doc, study_type)
        if not works:
            continue
        group = [
            {
                "score": interestingness(proj),
                "project_id": proj.id,
                "director": str(doc.id),
                "school": school,
                "title": proj.title,
                "description": _build_description(proj),
                "year": proj.year,
                "study_type": study_type,
                "keywords": proj.keywords,
            }
            for proj in works
        ]
        pool[school].append(group)
    for school in pool:
        pool[school].sort(key=lambda g: g[0]["score"], reverse=True)
    return pool


def proportional_quotas(total: int, proportions: dict[str, float]) -> dict[str, int]:
    """Per-school integer quotas summing to `total`, following `proportions`."""
    raw = {school: total * pct for school, pct in proportions.items()}
    floored = {school: int(q) for school, q in raw.items()}
    remainder = total - sum(floored.values())
    fracs = sorted(raw.items(), key=lambda x: -(x[1] - int(x[1])))
    for i in range(remainder):
        floored[fracs[i % len(fracs)][0]] += 1
    return floored


def _round_robin(
    groups: list[list[dict]], quota: int, used_ids: set[str]
) -> list[dict]:
    """Pick up to `quota` works, covering every PDI once before taking seconds.

    Works whose project_id is already in `used_ids` (e.g. a co-directed thesis
    claimed by another director) are skipped so each thesis appears only once.
    """
    iters = [iter(group) for group in groups]
    picks: list[dict] = []
    progress = True
    while len(picks) < quota and progress:
        progress = False
        for it in iters:
            if len(picks) >= quota:
                break
            for work in it:  # advance to this PDI's next unclaimed thesis
                if work["project_id"] in used_ids:
                    continue
                used_ids.add(work["project_id"])
                picks.append(work)
                progress = True
                break
    return picks


def sample_from_pool(
    pool: dict[str, list[list[dict]]],
    quotas: dict[str, int],
    total: int,
) -> list[dict]:
    """Fill each school's work quota (covering every PDI first), then top up any
    shortfall from remaining works of other schools to reach `total`."""
    selected: list[dict] = []
    used_ids: set[str] = set()
    for school, quota in quotas.items():
        groups = pool.get(school, [])
        chosen = _round_robin(groups, quota, used_ids)
        selected.extend(chosen)
        pdi_count = len(groups)
        work_count = sum(len(g) for g in groups)
        print(
            f"  {school}: quota {quota}, PDIs {pdi_count}, eligible works "
            f"{work_count}, selected {len(chosen)}"
        )

    shortfall = total - len(selected)
    if shortfall > 0:
        leftovers = [
            item
            for groups in pool.values()
            for group in groups
            for item in group
            if item["project_id"] not in used_ids
        ]
        leftovers.sort(key=lambda x: x["score"], reverse=True)
        topup = leftovers[:shortfall]
        selected.extend(topup)
        print(f"  Topped up {len(topup)} extra works to reach the target")

    return selected


def to_gold_record(item: dict, index: int) -> dict:
    """Convert a candidate dict to the GoldRecord schema."""
    return {
        "id": f"gold_{item['study_type'].lower()}_{index:04d}",
        "description": item["description"],
        "year": item["year"],
        "director": item["director"],
        "study_type": item["study_type"],
        "knowledge_area": item["school"],
        # Extra fields useful for debugging / error analysis.
        "_project_id": item["project_id"],
        "_title": item["title"],
        "_keywords": item["keywords"],
    }


def main() -> None:
    rids = load_researcher_ids()
    print(f"Loaded {len(rids)} researcher IDs")

    docs = [doc for rid in rids if (doc := load_enrichment(rid)) is not None]
    print(f"Loaded {len(docs)} enrichment documents\n")

    # PDI distribution per school (total professors) → sampling proportions.
    school_counts = Counter(
        s for doc in docs if (s := school_of(doc)) != "UNKNOWN"
    )
    total_pdis = sum(school_counts.values())
    proportions = {s: c / total_pdis for s, c in school_counts.items()}
    print("PDI distribution per school:")
    for school, count in school_counts.most_common():
        print(f"  {school}: {count} PDIs ({proportions[school]:.1%})")

    quotas = proportional_quotas(TARGET_TOTAL, proportions)
    print(f"\nTarget quotas per school: {quotas} (total {sum(quotas.values())})\n")

    out_dir = Path("evaluation")
    out_dir.mkdir(exist_ok=True)

    for study_type in ("TFG", "TFM"):
        print(f"=== Building {study_type} gold standard ===")
        pool = build_candidate_pool(docs, study_type)
        eligible_pdis = sum(len(groups) for groups in pool.values())
        eligible_works_total = sum(
            len(g) for groups in pool.values() for g in groups
        )
        print(
            f"  Eligible PDIs: {eligible_pdis}, eligible works: "
            f"{eligible_works_total}"
        )
        selected = sample_from_pool(pool, quotas, TARGET_TOTAL)
        distinct_pdis = len({item["director"] for item in selected})
        print(f"  Distinct PDIs covered: {distinct_pdis}")
        records = [to_gold_record(item, i + 1) for i, item in enumerate(selected)]

        out_path = out_dir / f"gold_standard_{study_type.lower()}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(records)} records -> {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
