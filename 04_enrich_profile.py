"""
Enrichment script for UPM Match.
This script builds a structured enrichment document for each PDI and persists it in MinIO.
The document is validated with Pydantic and contains the core profile information needed by the recommendation pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from loguru import logger

from dbs import get_graph_db, minio
from models import (
    ResearcherEnrichmentDocument,
    ResearcherLastYearProject,
    ResearcherPosition,
    ResearcherPublication,
    ResearcherSubject,
)
from utils.logging import setup_logging
from utils.researchers import ENRICHMENT_KEY_TEMPLATE, load_researcher_ids
from utils.text import canonicalize_topics

load_dotenv()

MULTISPACE_RE = re.compile(r"\s+")

# Only researchers linked (via :Works) to one of these Position nodes are persisted to MinIO.
# The rest are still loaded in Memgraph but excluded from the curated bucket.
TARGET_POSITION_IDS = {
    "loc_16086",  # ETSIT - Escuela Técnica Superior de Ingenieros de Telecomunicación
    "loc_16097",  # ETSII - Escuela Técnica Superior de Ingenieros Informáticos
    "loc_16331",  # ETSISI - Escuela Técnica Superior de Ingeniería de Sistemas Informáticos
    "loc_16380",  # ETSIST - Escuela Técnica Superior de Ingeniería y Sistemas de Telecomunicación
    # -------- #
    "loc_16408",  # ETSISI - Departamento de Sistemas Informáticos
    "loc_16409",  # ETSIST - Departamento de Ingeniería Telemática y Electrónica
    "loc_16410",  # ETSIST - Departamento de Lingüística Aplicada a la Ciencia y a la Tecnología
    "loc_16411",  # ETSIST - Departamento en Ingeniería Audiovisual y Comunicaciones
    "loc_16431",  # ETSII - Departamento de Arquitectura y Tecnología de Sistemas Informáticos
    "loc_16432",  # ETSII - Departamento de Inteligencia Artificial
    "loc_16433",  # ETSII - Departamento de Lenguajes e Ingeniería de Software
    "loc_16437",  # ETSIT - Departamento de Electrónica física, Ingeniería Eléctrica y Física Aplicada
    "loc_16438",  # ETSIT - Departamento de Ingeniería de Sistemas Telemáticos
    "loc_16439",  # ETSIT - Departamento de Ingeniería Electrónica
    "loc_16440",  # ETSIT - Departamento de Matemática Aplicada a Las Tecnologías de la Información y Las Comunicaciones
    "loc_16441",  # ETSIT - Departamento de Señales, Sistemas y Radiocomunicaciones
    "loc_16442",  # ETSIT - Departamento de Tecnología Fotónica y Bioingeniería
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the enrichment phase."""
    parser = argparse.ArgumentParser(
        description="Build and persist the researcher enrichment profile for UPM PDIs."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-i",
        "--id",
        type=int,
        help="Process only one researcher ID.",
    )
    group.add_argument(
        "-l",
        "--limit",
        type=int,
        help="Limit the number of researchers processed.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip researchers whose enrichment JSON already exists in MinIO.",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const="",
        default=None,
        metavar="PATH",
        help=(
            "Print the JSON document to stdout when used without a value, or "
            "write it to PATH.json when a value is provided."
        ),
    )
    parser.add_argument(
        "--logs",
        type=int,
        default=20,
        help="Numeric log level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR).",
    )
    return parser.parse_args()


def normalize_whitespace(value: str | None) -> str:
    """Collapse repeated whitespace and trim the result."""
    if not value:
        return ""
    return MULTISPACE_RE.sub(" ", str(value).replace("\xa0", " ")).strip()


def clean_text(value: Any) -> str | None:
    """Convert a value to a cleaned string or None if empty."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = normalize_whitespace(value)
        return cleaned if cleaned else None
    cleaned = normalize_whitespace(str(value))
    return cleaned if cleaned else None


def none_if_empty(value: Any) -> Any | None:
    """Convert empty strings and whitespace-only values to None."""
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = normalize_whitespace(value)
        return cleaned if cleaned else None
    return value


def extract_year(value: Any) -> int | None:
    """Extract a year from a value if possible."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.search(r"(\d{4})", text)
        if match:
            year = int(match.group(1))
            return year if year > 0 else None
    return None


def split_keywords(value: Any) -> list[str]:
    """Split keywords while preserving the stored spelling and order."""
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = re.split(r"[,;|/]", str(value))
    cleaned = [clean_text(item) for item in values]
    return [item.lower() for item in cleaned if item]


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Deduplicate while preserving the original order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = clean_text(value)
        if not cleaned:
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def infer_first_year(*year_groups: Iterable[int | None]) -> int | None:
    """Return the earliest positive year found across multiple collections."""
    years: list[int] = []
    for year_group in year_groups:
        for year in year_group:
            if isinstance(year, int) and year > 0:
                years.append(year)
    return min(years) if years else None


def build_researcher_row(
    researcher_id: int,
    first_name: str | None,
    last_name: str | None,
    first_year: int | None,
    topics: list[str],
    positions: list[ResearcherPosition],
    publications: list[ResearcherPublication],
    subjects: list[ResearcherSubject],
    last_year_projects: list[ResearcherLastYearProject],
) -> ResearcherEnrichmentDocument:
    """Assemble and validate the enrichment document."""
    return ResearcherEnrichmentDocument(
        id=f"pdi_{researcher_id}",
        first_name=first_name,
        last_name=last_name,
        first_year=first_year,
        topics=topics,
        positions=positions,
        publications=publications,
        subjects=subjects,
        last_year_projects=last_year_projects,
        generated_at=datetime.now(timezone.utc),
    )


def researcher_has_target_position(graph, researcher_id: int) -> bool:
    """Return True if the researcher is linked to one of the target Position nodes."""
    graph_id = f"pdi_{researcher_id}"
    rows = graph.query(
        """
        MATCH (r:Pdi {id: $researcher_id})-[:Works]->(pos:Position)
        WHERE pos.id IN $position_ids
        RETURN count(pos) AS matches
        """,
        {"researcher_id": graph_id, "position_ids": list(TARGET_POSITION_IDS)},
    )
    return bool(rows) and (rows[0].get("matches") or 0) > 0


def query_researcher_records(graph, researcher_id: int) -> dict[str, Any]:
    """Load the researcher and its related nodes from Memgraph."""
    graph_id = f"pdi_{researcher_id}"

    researcher_rows = graph.query(
        """
        MATCH (r:Pdi {id: $researcher_id})
        RETURN r.id AS id,
               r.first_name AS first_name,
               r.last_name AS last_name,
               r.topics AS topics
        """,
        {"researcher_id": graph_id},
    )
    if not researcher_rows:
        return {}

    publication_rows = graph.query(
        """
        MATCH (r:Pdi {id: $researcher_id})-[:Authorship]->(p:Publication)
        OPTIONAL MATCH (p)<-[:Authorship]-(co:Pdi)
        WHERE co.id <> $researcher_id
        WITH p, collect(DISTINCT co.id) AS coauthors
        RETURN p.id AS id,
               p.name AS title,
               p.description AS abstract,
               p.type as type,
               p.year AS year,
               p.source AS source,
               p.keywords AS keywords,
               coauthors AS coauthorship
        ORDER BY coalesce(p.year, 0) DESC, coalesce(p.name, p.title) ASC
        """,
        {"researcher_id": graph_id},
    )

    subject_rows = graph.query(
        """
        MATCH (r:Pdi {id: $researcher_id})-[t:Teaching]->(s:Subject)
        WHERE s.type IN ["Docencia Oficial Nacional", "Docencia Oficial Internacional"]
        RETURN s.name AS name,
               t.start_date AS start_date,
               t.end_date AS end_date
        ORDER BY coalesce(start_date, "0000-01-01") ASC,
                 coalesce(end_date, "9999-12-31") ASC,
                 name ASC
        """,
        {"researcher_id": graph_id},
    )

    position_rows = graph.query(
        """
        MATCH (r:Pdi {id: $researcher_id})-[w:Works]->(pos:Position)
        WHERE coalesce(pos.entity, "") <> "Área de Conocimiento"
        RETURN pos.id AS id,
               pos.entity AS entity,
               pos.name AS name,
               w.role AS role,
               w.since AS since
        ORDER BY since ASC
        """,
        {"researcher_id": graph_id},
    )

    last_year_project_rows = graph.query(
        """
        MATCH (r:Pdi {id: $researcher_id})-[:Tutoring]->(p:LastYearProject)
        WHERE p.type IN ["TFG", "TFM"]
        RETURN p.id AS id,
               p.name AS title,
               p.description AS abstract,
               p.type AS type,
               p.keywords AS keywords,
               p.year AS year
        ORDER BY coalesce(year, 0) DESC, title ASC
        """,
        {"researcher_id": graph_id},
    )

    return {
        "researcher": researcher_rows[0],
        "publications": publication_rows,
        "subjects": subject_rows,
        "positions": position_rows,
        "last_year_projects": last_year_project_rows,
    }


def build_positions(rows: list[dict[str, Any]]) -> list[ResearcherPosition]:
    """Validate researcher positions."""
    positions: list[ResearcherPosition] = []
    for row in rows:
        position = ResearcherPosition.model_validate(
            {
                "id": none_if_empty(row.get("id")) or "",
                "name": none_if_empty(row.get("name")),
                "entity": none_if_empty(row.get("entity")),
                "role": none_if_empty(row.get("role")),
                "since": extract_year(row.get("since")),
            }
        )
        positions.append(position)
    return positions


def build_publications(rows: list[dict[str, Any]]) -> list[ResearcherPublication]:
    """Validate researcher publications."""
    publications: list[ResearcherPublication] = []
    for row in rows:
        publication = ResearcherPublication.model_validate(
            {
                "id": none_if_empty(row.get("id")) or "",
                "title": none_if_empty(row.get("title")),
                "abstract": none_if_empty(row.get("abstract")),
                "type": none_if_empty(row.get("type")),
                "year": extract_year(row.get("year")),
                "source": none_if_empty(row.get("source")),
                "keywords": canonicalize_topics(split_keywords(row.get("keywords"))),
                "coauthorship": [
                    author_id
                    for author_id in unique_preserve_order(
                        row.get("coauthorship") or []
                    )
                    if author_id
                ],
            }
        )
        publications.append(publication)
    return publications


def build_subjects(rows: list[dict[str, Any]]) -> list[ResearcherSubject]:
    """Validate researcher subjects."""
    subjects: list[ResearcherSubject] = []
    for row in rows:
        subject = ResearcherSubject.model_validate(
            {
                "name": none_if_empty(row.get("name")),
                "start_year": extract_year(row.get("start_date")),
                "end_year": extract_year(row.get("end_date")),
            }
        )
        subjects.append(subject)
    return subjects


def build_last_year_projects(
    rows: list[dict[str, Any]],
) -> list[ResearcherLastYearProject]:
    """Validate researcher tutoring records."""
    projects: list[ResearcherLastYearProject] = []
    for row in rows:
        project = ResearcherLastYearProject.model_validate(
            {
                "id": none_if_empty(row.get("id")) or "",
                "title": none_if_empty(row.get("title")),
                "abstract": none_if_empty(row.get("abstract")),
                "type": none_if_empty(row.get("type")),
                "keywords": canonicalize_topics(split_keywords(row.get("keywords"))),
                "year": extract_year(row.get("year")),
            }
        )
        projects.append(project)
    return projects


def build_document(
    researcher_id: int,
    subgraph: dict[str, Any],
) -> ResearcherEnrichmentDocument:
    """Build the final Pydantic document for a researcher."""
    researcher_row = subgraph["researcher"]
    first_name = researcher_row.get("first_name")
    last_name = researcher_row.get("last_name")
    topics = canonicalize_topics(researcher_row.get("topics") or [])
    positions = build_positions(subgraph.get("positions", []))
    publications = build_publications(subgraph.get("publications", []))
    subjects = build_subjects(subgraph.get("subjects", []))
    last_year_projects = build_last_year_projects(
        subgraph.get("last_year_projects", [])
    )
    first_year = infer_first_year(
        (position.since for position in positions),
        (publication.year for publication in publications),
        (subject.start_year for subject in subjects),
        (project.year for project in last_year_projects),
    )
    return build_researcher_row(
        researcher_id=researcher_id,
        first_name=first_name,
        last_name=last_name,
        first_year=first_year,
        topics=topics,
        positions=positions,
        publications=publications,
        subjects=subjects,
        last_year_projects=last_year_projects,
    )


def persist_document(
    researcher_id: int, document: ResearcherEnrichmentDocument
) -> None:
    """Persist the validated document in MinIO."""
    object_name = ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
    if not minio.upload_json(
        minio.CURATED_BUCKET, document.model_dump(mode="json"), object_name
    ):
        raise RuntimeError(f"Failed to persist enrichment document: {object_name}")


def print_document(document: ResearcherEnrichmentDocument) -> None:
    """Print the validated document in JSON form."""
    print(json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2))


def write_document(path_value: str, document: ResearcherEnrichmentDocument) -> Path:
    """Write the validated document to a JSON file and return the path used."""
    output_path = Path(path_value)
    if output_path.suffix.lower() != ".json":
        output_path = output_path.with_suffix(".json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def process_researcher(
    graph,
    researcher_id: int,
    skip_existing: bool = False,
    debug: str | None = None,
) -> bool:
    """Run the enrichment for one researcher."""
    object_name = ENRICHMENT_KEY_TEMPLATE.format(researcher_id=researcher_id)
    if skip_existing and minio.object_exists(minio.CURATED_BUCKET, object_name):
        logger.info("Skipping {} because {} already exists", researcher_id, object_name)
        return False

    subgraph = query_researcher_records(graph, researcher_id)
    if not subgraph:
        logger.warning("Researcher {} not found in Memgraph", researcher_id)
        return False

    # In production mode only persist researchers tied to the target Position nodes.
    # Debug mode bypasses the filter so any profile can be inspected.
    if debug is None and not researcher_has_target_position(graph, researcher_id):
        logger.info(
            "Skipping {} — no relationship with target Position nodes", researcher_id
        )
        return False

    document = build_document(researcher_id, subgraph)

    if debug is not None:
        if debug == "":
            print_document(document)
            logger.info(
                "Debug output generated for {} with {} publications and {} projects",
                researcher_id,
                len(document.publications),
                len(document.last_year_projects),
            )
        else:
            output_path = write_document(debug, document)
            logger.info(
                "Debug output written to {} for {} with {} publications and {} projects",
                output_path,
                researcher_id,
                len(document.publications),
                len(document.last_year_projects),
            )
    else:
        persist_document(researcher_id, document)
        logger.info(
            "Persisted enrichment for {} with {} publications and {} projects",
            researcher_id,
            len(document.publications),
            len(document.last_year_projects),
        )
    return True


def resolve_researcher_ids(args: argparse.Namespace) -> list[int]:
    """Resolve the researchers to process from CLI arguments."""
    researcher_ids = load_researcher_ids()
    if args.id is not None:
        researcher_ids = [
            researcher_id
            for researcher_id in researcher_ids
            if researcher_id == args.id
        ]
    if args.limit is not None:
        researcher_ids = researcher_ids[: args.limit]
    return researcher_ids


def main() -> None:
    """Run the enrichment pipeline."""
    args = parse_args()

    setup_logging("enrich_profile", level=args.logs)
    minio.ensure_bucket(minio.CURATED_BUCKET)

    researcher_ids = resolve_researcher_ids(args)
    if not researcher_ids:
        logger.warning("No researchers to process")
        return

    graph = get_graph_db("memgraph")
    graph.ensure_collections()

    processed = 0
    skipped = 0
    failed = 0

    if args.id is not None:
        logger.info("Processing a single researcher: {}", args.id)

    for researcher_id in researcher_ids:
        try:
            if process_researcher(
                graph,
                researcher_id,
                skip_existing=args.skip_existing,
                debug=args.debug,
            ):
                processed += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            logger.error("Failed to process {}: {}", researcher_id, exc)

    logger.info(
        "Finished enrichment. processed={} skipped={} failed={}",
        processed,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
