"""Read-only Memgraph access for the public researcher profile.

Loads the academic-trajectory data shown in the profile explorer (/search):
supervised works (TFG/TFM/theses), subjects taught and positions held. All
lookups degrade gracefully — if Memgraph is unreachable or the researcher has
no graph data, the profile simply renders without these sections.
"""

from __future__ import annotations

from loguru import logger

# Lazily-created backend. False means "tried and failed" (missing env or
# unreachable), so we don't retry the connection on every profile load.
_graph = None


def _backend():
    global _graph
    if _graph is None:
        try:
            from dbs.graph.memgraph import MemgraphBackend  # noqa: PLC0415 — lazy: requires MEMGRAPH_* env

            _graph = MemgraphBackend()
        except Exception:  # noqa: BLE001 — degradar con elegancia sin grafo
            logger.exception("Memgraph unavailable; profile graph sections disabled")
            _graph = False
    return _graph or None


def _merge_period(current: dict, start: str | None, end: str | None) -> None:
    """Widen a subject's teaching period with another edge's start/end dates."""
    if start and (not current["start"] or start < current["start"]):
        current["start"] = start
    if end and (not current["end"] or end > current["end"]):
        current["end"] = end


def _year(date: str | None) -> int | None:
    """Extract the year from a YYYY-MM-DD string."""
    if date and len(date) >= 4 and date[:4].isdigit():
        return int(date[:4])
    return None


def load_graph_profile(researcher_id: int) -> dict | None:
    """Load supervised works, subjects and positions for one researcher.

    :param researcher_id: numeric portal id (e.g. 308607).
    :return: {"projects": [...], "subjects": [...], "positions": [...]} or None
        when the graph is unavailable. Lists may be empty (no data ≠ no graph).
    """
    db = _backend()
    if db is None:
        return None

    pdi_id = f"pdi_{researcher_id}"

    # Supervised final works and theses, most recent first.
    projects = [
        {
            "title": (row.get("title") or "").strip() or "(sin título)",
            "year": row.get("year"),
            "type": (row.get("type") or "").strip() or None,
            "oa_url": row.get("oa_url") or None,
        }
        for row in db.query(
            """
            MATCH (:Pdi {id: $id})-[:Tutoring]->(t:LastYearProject)
            RETURN t.name AS title, t.year AS year, t.type AS type, t.oa_id AS oa_url
            ORDER BY t.year DESC, t.name
            """,
            {"id": pdi_id},
        )
    ]

    # Subjects taught: one entry per subject name, merging the periods of the
    # (possibly several) Teaching edges into the widest [start, end] range.
    by_name: dict[str, dict] = {}
    for row in db.query(
        """
        MATCH (:Pdi {id: $id})-[e:Teaching]->(s:Subject)
        RETURN s.name AS name, e.start_date AS start, e.end_date AS end
        """,
        {"id": pdi_id},
    ):
        name = (row.get("name") or "").strip()
        if not name:
            continue
        entry = by_name.setdefault(
            name.lower(), {"name": name, "start": None, "end": None}
        )
        _merge_period(entry, row.get("start"), row.get("end"))
    subjects = [
        {"name": s["name"], "from": _year(s["start"]), "to": _year(s["end"])}
        for s in sorted(
            by_name.values(), key=lambda s: (s["end"] or "", s["start"] or ""), reverse=True
        )
    ]

    # Positions held (department, research group, centre...).
    positions = [
        {
            "name": (row.get("name") or "").strip(),
            "entity": (row.get("entity") or "").strip() or None,
            "role": (row.get("role") or "").strip() or None,
            "since": _year(row.get("since")),
        }
        for row in db.query(
            """
            MATCH (:Pdi {id: $id})-[e:Works]->(x:Position)
            RETURN x.name AS name, x.entity AS entity, e.role AS role, e.since AS since
            """,
            {"id": pdi_id},
        )
        if (row.get("name") or "").strip()
    ]

    return {"projects": projects, "subjects": subjects, "positions": positions}
