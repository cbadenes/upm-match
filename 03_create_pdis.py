"""
This script processes the raw HTML files of researchers' profiles and activities to extract structured information about their personal data, current positions, external IDs, research subjects and publications.
It uses Scrapy Selectors to parse the HTML content and Polars for data manipulation.
The extracted information is then transformed into validated Pydantic models.
The script also includes functions to clean and normalize publication descriptions to ensure consistency and quality of the extracted data.
Finally, the structured data is stored in a graph database for further analysis and querying.

For each researcher, it is generated these nodes and edges:
- `Researcher` node with personal data and external IDs.
- `Publication` nodes for each publication, connected to the `Researcher` node with "Authorship" edges.
- `Position` nodes for each current position, connected to the `Researcher` node with "Works" edges.
- `Subject` nodes for each research topic, connected to the `Researcher` node with "Teaching" edges.

The information is extracted from the two HTML files downloaded in the previous step (`profile.html` and `activity.html`) and is cleaned and normalized to ensure quality and consistency of the data before being stored in the graph database.
The HTML files can be loaded from the local filesystem or directly from MinIO, depending on the configuration and availability of the files.
The data is processed in a way that handles missing or inconsistent information gracefully, ensuring that the resulting graph database contains accurate and meaningful representations of the researchers and their activities.
The results can be stored to Memgraph (default option), ArangoDB or MinIO/local JSON files, depending on the configuration.

Use:
> uv run 03_create_pdis.py
> uv run 03_create_pdis.py --id 308607
> uv run 03_create_pdis.py --limit 1000 --html_source minio
> uv run 03_create_pdis.py --skip-existing
> uv run 03_create_pdis.py --logs 30 --progress-every 250
"""

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import polars as pl
from loguru import logger
from scrapy import Selector

from dbs import GraphDataBase, get_graph_db, minio
from models import (
    Authorship,
    LastYearProject,
    Position,
    Publication,
    Researcher,
    Subject,
    Teaching,
    Tutoring,
    Works,
)
from utils.logging import setup_logging
from utils.researchers import load_researcher_ids_with_signatures
from utils.text import (
    canonicalize_topics,
    choose_keyword_source,
    choose_publication_description,
    clean_description,
    normalize_topic,
    split_publication_keywords,
)

PUBLICATIONS_REGISTRY = {}  # Per-researcher publication COUNTS (not the objects)
SUBJECTS_REGISTRY = {}  # Per-researcher subject COUNTS (not the objects)
LYP_REGISTRY = {}  # Per-researcher LastYearProject COUNTS (not the objects)


def normalize_last_name(value: Optional[str], fallback: str = "sin_apellido") -> str:
    """Build a filesystem/object-safe token from a last name."""
    raw = str(value or fallback).strip()
    token = raw.split()[0] if raw else fallback
    return re.sub(r"[^A-Za-z0-9_-]", "_", token)


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to mm:ss or hh:mm:ss."""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def load_html_file(
    researcher_id: int,
    file_type: str,
    html_source: str = "local",
    raw_dir: Path = Path("data/raw"),
    minio_prefix: str = "raw",
) -> Optional[str]:
    """Load HTML file content from local filesystem or MinIO.

    :param researcher_id: The ID of the researcher.
    :param file_type: Type of HTML file ('main' or 'ajax').
    :param html_source: Source to read from ('local' or 'minio'). Defaults to 'local'.
    :param raw_dir: Local directory containing raw HTML files (used if html_source='local').
    :param minio_prefix: MinIO prefix for object keys (used if html_source='minio').
    :return: File content as string, or None if file not found or error occurs.
    """
    if html_source == "local":
        file_path = raw_dir / f"{researcher_id}_{file_type}.html"
        try:
            if file_path.exists():
                return file_path.read_text(encoding="utf-8")
            else:
                logger.debug("Local HTML file not found: {}", file_path)
                return None
        except Exception as exc:
            logger.debug("Error reading local HTML file {}: {}", file_path, exc)
            return None

    elif html_source == "minio":
        if file_type == "main":
            object_key = f"{minio_prefix}/{researcher_id}/profile.html"
        elif file_type == "ajax":
            object_key = f"{minio_prefix}/{researcher_id}/activity.html"
        else:
            object_key = f"{minio_prefix}/{researcher_id}_{file_type}.html"  # if file_type is something else, fallback to old naming convention
        try:
            content = minio.read_file(minio.RAW_BUCKET, object_key)
            if content is not None:
                return content
            else:
                logger.debug("MinIO HTML object not found: {}", object_key)
                return None
        except Exception as exc:
            logger.debug("Error reading MinIO HTML file {}: {}", object_key, exc)
            return None

    else:
        logger.error(
            "Unknown html_source: {}. Must be 'local' or 'minio'.", html_source
        )
        return None


def _parse_activity_json(ajax_html: str) -> list[dict] | None:
    """Extract the raw activity JSON list from the AJAX HTML content.

    :param ajax_html: The AJAX HTML content as a string.
    :return: The raw list of activity records, or None if not found or unparseable.
    """
    match = re.search(r"var dataSet\s*=\s*(\[.*\]);", ajax_html, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        try:
            json_str = match.group(1).rsplit("]", 1)[0] + "]"
            return json.loads(json_str)
        except Exception:
            return None


def load_ajax_dataSet(ajax_html: str) -> pl.DataFrame:
    """Load and clean the dataset from AJAX HTML content string.

    :param ajax_html: The AJAX HTML content as a string.
    :return: A cleaned Polars DataFrame with the relevant activity data extracted from the AJAX HTML.
    """
    activity_data = _parse_activity_json(ajax_html)

    if not activity_data:
        return pl.DataFrame()

    all_fields = set().union(*(record.keys() for record in activity_data))
    normalized_records = [
        {f: record.get(f) for f in all_fields} for record in activity_data
    ]

    df = pl.DataFrame(normalized_records, infer_schema_length=len(normalized_records))

    mapeo_columnas = {
        "rfTitle": "ACTIVIDAD",
        "dsDescription": "DESCRIPTION",
        "rfAuthor": "AUTORES",
        "rfKeyword": "KEYWORDS",
        "rfStartDate": "FECHA INICIO",
        "rfEndDate": "FECHA FIN",
        "rfHash": "IDENTIFICADOR",
        # "stOpenAccess": "OPENACCESS",
        "dtWorkform": "TIPO",
        "dtType": "SUBTIPO",
        "rfSource": "FUENTE",
        "rfValidationAgency": "VALIDACIÓN",
        "rfSubject": "CATEGORÍAS",
        # "rfLinkName": "FUENTE",
        # "rfQuality11": "IF SJR",
        # "rfQuality17": "CITAS EUROPEPMC",
        # "rfQuality18": "CITAS INSPIRE",
        # "rfQuality19": "CITAS SCHOLAR",
        # "rfQuality22": "Q SJR",
        # "rfRef": "CITACIÓN",
        # "idQuality63": "PIDs",
        "rfRefUrl": "URL",
        "dsDescription5": "DESCRIPCION TFT",
        "idItem": "PORTAL_UPM_ID",
    }

    columnas_presentes = [c for c in mapeo_columnas.keys() if c in df.columns]
    if not columnas_presentes or "dtWorkform" not in columnas_presentes:
        return pl.DataFrame()

    mapeo_final = {k: mapeo_columnas[k] for k in columnas_presentes}

    df_final = df.select(columnas_presentes).rename(mapeo_final)

    keyword_like_columns = {
        column for column in ["KEYWORDS", "CATEGORÍAS"] if column in df_final.columns
    }
    string_columns = [
        column
        for column, dtype in df_final.schema.items()
        if dtype == pl.Utf8 and column != "URL"
    ]

    cleaned_expressions = []
    for column in string_columns:
        expression = pl.col(column).str.replace_all(r"[\r\n]+", "; ")
        if column not in keyword_like_columns:
            expression = expression.str.replace_all(r";", ",")

        cleaned_expressions.append(
            expression.str.replace_all(r"\s+", " ")
            .str.strip_chars('"')
            .str.strip_chars()
            .alias(column)
        )

    if cleaned_expressions:
        df_final = df_final.with_columns(cleaned_expressions)

    if "URL" in df_final.columns:
        df_final = df_final.with_columns(
            pl.col("URL").map_elements(
                lambda s: ", ".join(re.findall(r"https?://\S+", s)) if s else s,
                return_dtype=pl.String,
            )
        )

    df_final = df_final.sort(
        [c for c in ["TIPO", "SUBTIPO", "FECHA INICIO"] if c in df_final.columns],
        descending=False,
    )

    # print("!!!!!!!!!" * 10)
    # print(f"{save=}")
    # if save is True:
    #     df_final.write_csv(f"ajax_{ID}_limpio.csv")

    return df_final


def load_personal_data(
    sel: Selector,
    rf_signature: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Load personal data of the researcher from the main HTML selector.

    :param sel: Scrapy Selector object containing the parsed main HTML of the researcher's profile.
    :param rf_signature: Optional "APELLIDOS, NOMBRE" value from the CSV to split names.
    :return: A tuple containing the full name, first name, last name, and email of the researcher.
    """
    title_text = sel.css("title::text").get("")
    full_name = (
        title_text.split(" - ")[0].strip().title() if " - " in title_text else None
    )

    first_name, last_name = None, None
    if rf_signature and "," in rf_signature:
        parts = rf_signature.split(",", 1)
        last_name = parts[0].strip().title()
        first_name = parts[1].strip().title()

    email = sel.xpath('//a[contains(@class,"fa-envelope")]/@data-original-title').get()
    return full_name, first_name, last_name, email


def build_position_key(
    title: Optional[str],
    entity: Optional[str],
    source_id: Optional[str],
) -> str:
    """Build a stable key for a current position."""
    if source_id:
        return f"loc_{source_id}"

    canonical = "|".join(part or "" for part in [title, entity])
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    return f"loc_{digest}"


def load_current_positions(sel: Selector) -> list[Position]:
    """Load the current positions of the researcher from the main HTML selector.

    :param sel: Scrapy Selector object containing the parsed main HTML of the researcher's profile.
    :return: A list of validated Position models with capitalized node names.
    """
    # TODO: Arreglar si aparecen en dos sitios iguales (en el mismo departamento en el que trabaja se ha vuelto catedrático y aparece el mismo dos veces)
    positions = []
    for event in sel.xpath(
        '//div[contains(@class,"ipublic-panel")]//div[@class="event"]'
    ):
        title = event.xpath("normalize-space(./a/text())").get()
        entity = event.xpath("normalize-space(./div[1]/text())").get()
        role = event.xpath("normalize-space(./div[2]/text())").get()
        since = event.xpath("normalize-space(./div[3]/text())").get()
        source_id = event.xpath("./a/@href").re_first(r"(\d+)$")

        position_id = build_position_key(title, entity, source_id)

        # Convert to validated Pydantic model with capitalized node names
        position = Position(
            id=f"{position_id}",
            name=title or entity,
            entity=entity,
            role=role,
            since=since,
            source_id=source_id,
        )
        positions.append(position)

    return positions


def load_external_ids(sel: Selector) -> dict[str, str | None]:
    """Load the external IDs of the researcher from the main HTML selector.

    :param sel: Scrapy Selector object containing the parsed main HTML of the researcher's profile.
    :return: A dictionary where keys are the types of external IDs (e.g., 'orcid_id', 'scopus_id') and values are the corresponding IDs extracted from the profile, or None if not found.
    """

    def extract_id_from_title(sel, href_fragment) -> Optional[str]:
        """Extract the ID from the 'data-original-title' attribute of an <a> tag that contains a specific href fragment.

        :param sel: Scrapy Selector object containing the parsed main HTML of the researcher's profile.
        :param href_fragment: A substring to identify the correct <a> tag (e.g., 'orcid.org', 'scopus.com').
        :return: The extracted ID from the title, or None if not found or if the title format is unexpected.
        """
        title = sel.xpath(
            f'//a[contains(@href, "{href_fragment}")]/@data-original-title'
        ).get()
        if title and " - " in title:
            return title.split(" - ", 1)[1].strip()
        return None

    ids_researcher = {
        # "researcher": rf_signature,
        # "id_interno": ID,
        # "email": sel.xpath(
        #     '//a[contains(@class,"fa-envelope")]/@data-original-title'
        # ).get(),
        "personal_web": sel.xpath(
            '//a[contains(@class,"fa-external-link") or @title[starts-with(., "http")]]/@href'
        ).get(),
        # "orcid_url": sel.xpath('//a[contains(@href,"orcid.org")]/@href').get(),
        "orcid_id": extract_id_from_title(sel, "orcid.org"),
        # "researcher_id_url": sel.xpath(
        #     '//a[contains(@href,"webofscience.com")]/@href'
        # ).get(),
        "researcher_id": extract_id_from_title(sel, "webofscience.com"),
        # "scopus_url": sel.xpath('//a[contains(@href,"scopus.com")]/@href').get(),
        "scopus_id": extract_id_from_title(sel, "scopus.com"),
        # "google_scholar_url": sel.xpath(
        #     '//a[contains(@href,"scholar.google")]/@href'
        # ).get(),
        "google_scholar_id": extract_id_from_title(sel, "scholar.google"),
        # "researchgate_url": sel.xpath(
        #     '//a[contains(@href,"researchgate.net")]/@href'
        # ).get(),
        "researchgate_id": extract_id_from_title(sel, "researchgate.net"),
        # "dialnet_url": sel.xpath(
        #     '//a[contains(@href,"dialnet.unirioja.es")]/@href'
        # ).get(),
        "dialnet_id": extract_id_from_title(sel, "dialnet.unirioja.es"),
        # "mendeley_url": sel.xpath('//a[contains(@class,"bd-mendeley")]/@href').get(),
        "mendeley_id": extract_id_from_title(sel, "mendeley.com"),
        # "linkedin_url": sel.xpath('//a[contains(@class,"bd-linkedin")]/@href').get(),
        "linkedin_id": extract_id_from_title(sel, "linkedin.com"),
    }

    ids_researcher = {
        k: (None if v in ("", "https://", "http://") else v)
        for k, v in ids_researcher.items()
    }
    return ids_researcher


def build_authorship_edges(
    pdi_id: str, publications: list[Publication]
) -> list[Authorship]:
    """Generate validated Authorship edge models for publications.

    :param pdi_id: The PDI identifier (e.g., "pdi_307844").
    :param publications: A list of Publication instances.
    :return: A list of validated Authorship edge models with capitalized node names.
    """
    edges = []
    for pub in publications:
        # Extract base ID from Publication (which may include prefix)
        base_pub_id = pub.id.split("_", 1)[-1] if "_" in pub.id else pub.id
        base_pdi_id = pdi_id.split("_", 1)[-1] if "_" in pdi_id else pdi_id

        edge = Authorship(
            id=f"authorship_{base_pdi_id}_{base_pub_id}",
            _from=f"Pdi/{pdi_id}",
            _to=f"Publication/{pub.id}",
        )
        edges.append(edge)

    return edges


def build_subject_key(name: str, subtype: str) -> str:
    """Build a deterministic key for a subject derived from its name and subtype."""
    canonical = f"{name}|{subtype}"
    digest = hashlib.sha1(canonical.encode()).hexdigest()[:8]
    return f"subj_{digest}"


def build_subjects(df_ajax: pl.DataFrame) -> list[Subject]:
    """Build validated Subject models from AJAX dataset.

    :param df_ajax: Cleaned DataFrame from AJAX data.
    :return: List of validated Subject models.
    """
    if df_ajax.is_empty() or "TIPO" not in df_ajax.columns:
        return []

    required_columns = {"ACTIVIDAD", "SUBTIPO", "FECHA INICIO", "FECHA FIN"}
    if not required_columns.issubset(set(df_ajax.columns)):
        return []

    docencia = df_ajax.filter(pl.col("TIPO") == "Docencia Impartida").with_columns(
        [
            pl.col("ACTIVIDAD")
            .fill_null("")
            .str.strip_chars()
            .str.replace_all('"', "")
            .str.replace_all(r"\s+", " ")
            .str.to_lowercase()
            .str.to_titlecase()
            .alias("ACTIVIDAD"),
            pl.col("SUBTIPO")
            .fill_null("")
            .str.strip_chars()
            .str.replace_all('"', "")
            .str.replace_all(r"\s+", " ")
            .str.to_lowercase()
            .str.to_titlecase()
            .alias("SUBTIPO"),
            pl.col("FECHA INICIO")
            .cast(pl.Utf8, strict=False)
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("FECHA INICIO"),
            pl.col("FECHA FIN")
            .cast(pl.Utf8, strict=False)
            .str.to_date("%Y-%m-%d", strict=False)
            .alias("FECHA FIN"),
        ]
    )

    if docencia.is_empty():
        return []

    maxend_date = docencia.select(pl.col("FECHA FIN").max()).item()

    if maxend_date is not None:
        docencia = docencia.with_columns(
            pl.when(
                (pl.col("SUBTIPO") == "Docencia Oficial Nacional")
                & pl.col("FECHA INICIO").is_not_null()
                & pl.col("FECHA FIN").is_null()
            )
            .then(pl.lit(maxend_date, dtype=pl.Date))
            .otherwise(pl.col("FECHA FIN"))
            .alias("FECHA FIN")
        )

    subjects_dicts = (
        docencia.group_by(["ACTIVIDAD", "SUBTIPO"])
        .agg(
            [
                pl.col("FECHA INICIO").min().alias("FECHA INICIO"),
                pl.col("FECHA FIN").max().alias("FECHA FIN"),
            ]
        )
        .with_columns(
            pl.struct(["ACTIVIDAD", "SUBTIPO"])
            .map_elements(
                lambda x: build_subject_key(x["ACTIVIDAD"], x["SUBTIPO"]),
                return_dtype=pl.String,
            )
            .alias("id")
        )
        .select(
            [
                "id",
                pl.col("ACTIVIDAD").alias("name"),
                pl.col("SUBTIPO").alias("type"),
                pl.col("FECHA INICIO").dt.strftime("%Y-%m-%d").alias("start_date"),
                pl.col("FECHA FIN").dt.strftime("%Y-%m-%d").alias("end_date"),
            ]
        )
        .sort(["name", "start_date"])
        .to_dicts()
    )

    # Convert to validated Pydantic models with capitalized node names
    subjects = []
    for s_dict in subjects_dicts:
        subject = Subject(
            id=f"{s_dict['id']}",
            name=s_dict["name"],
            type=s_dict["type"],
            start_date=s_dict.get("start_date"),
            end_date=s_dict.get("end_date"),
        )
        subjects.append(subject)

    return subjects


def build_last_year_project(df_ajax: pl.DataFrame) -> list[LastYearProject]:
    """Build validated LastYearProject models from AJAX dataset.

    :param df_ajax: Cleaned DataFrame from AJAX data.
    :return: List of validated LastYearProject models with capitalized node names.
    """
    if df_ajax.is_empty() or "TIPO" not in df_ajax.columns:
        return []

    df_lyp = (
        df_ajax.filter(pl.col("TIPO") == "Dirección de Tesis, TFG, TFM, etc.")
        .rename(
            {
                "PORTAL_UPM_ID": "portal_upm_id",
                "ACTIVIDAD": "title",
                "AUTORES": "authors",
                "KEYWORDS": "keywords",
                "FECHA INICIO": "start_date",
                "IDENTIFICADOR": "oa_id",
                "SUBTIPO": "type",
                "DESCRIPCION TFT": "description",
            }
        )
        .with_columns(
            pl.format("tft_{}", pl.col("portal_upm_id")).alias("id"),
            pl.col("start_date")
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .dt.year()
            .alias("year"),
        )
    )

    last_year_projects = []
    for record in df_lyp.to_dicts():
        title = str(record.get("title") or "").strip()
        portal_upm_id = record.get("portal_upm_id")
        if not title or not portal_upm_id:
            logger.warning(
                "Skipping LastYearProject with empty title or portal_upm_id (portal_upm_id={})",
                portal_upm_id,
            )
            continue

        raw_oa_id = record.get("oa_id")
        oa_id = str(raw_oa_id).strip() if raw_oa_id else None
        description = clean_description(record.get("description")) or ""
        authors = [
            a.strip() for a in str(record.get("authors", "")).split(",") if a.strip()
        ]
        keywords = canonicalize_topics(
            split_publication_keywords(choose_keyword_source(record))
        )

        project = LastYearProject(
            id=f"{record['id']}",
            oa_id=oa_id,
            title=title,
            description=description,
            authors=authors,
            keywords=keywords,
            year=int(record.get("year") or 0),
            type=record.get("type", ""),
        )
        last_year_projects.append(project)
    return last_year_projects


def build_teaching_edges(pdi_id: str, subjects: list[Subject]) -> list[Teaching]:
    """Generate validated Teaching edge models for subjects.

    :param pdi_id: The PDI identifier.
    :param subjects: A list of Subject instances.
    :return: A list of validated Teaching edge models with temporal metadata.
    """
    edges = []

    for subject in subjects:
        base_pdi_id = pdi_id.split("_", 1)[-1] if "_" in pdi_id else pdi_id

        edge = Teaching(
            id=f"teaching_{base_pdi_id}_{subject.id}",
            _from=f"Pdi/{pdi_id}",
            _to=f"Subject/{subject.id}",
            start_date=subject.start_date,
            end_date=subject.end_date,
        )
        edges.append(edge)

    return edges


def build_tutoring_edges(
    pdi_id: str, last_year_projects: list[LastYearProject]
) -> list[Tutoring]:
    """Generate validated Tutoring edge models for last year projects."""
    edges = []

    for lyp in last_year_projects:
        base_pdi_id = pdi_id.split("_", 1)[-1] if "_" in pdi_id else pdi_id

        edge = Tutoring(
            id=f"tutoring_{base_pdi_id}_{lyp.id}",
            _from=f"Pdi/{pdi_id}",
            _to=f"LastYearProject/{lyp.id}",
        )
        edges.append(edge)

    return edges


def build_works_edges(pdi_id: str, positions: list[Position]) -> list[Works]:
    """Generate validated Works edge models for positions.

    :param pdi_id: The PDI identifier.
    :param positions: A list of Position instances.
    :return: A list of validated Works edge models with role and temporal info.
    """
    edges = []

    for position in positions:
        base_pdi_id = pdi_id.split("_", 1)[-1] if "_" in pdi_id else pdi_id

        edge = Works(
            id=f"works_{base_pdi_id}_{position.id}",
            _from=f"Pdi/{pdi_id}",
            _to=f"Position/{position.id}",
            role=position.role,
            since=position.since,
        )
        edges.append(edge)

    return edges


def extract_topic_counts(html: str) -> dict[str, int]:
    """Extract the raw topic counts from the embedded `data1` JSON blob."""
    match = re.search(r"var data1\s*=\s*JSON\.parse\('(.*?)'\);", html, re.DOTALL)
    if not match:
        return {}

    try:
        data1 = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        logger.warning("Error parsing data1: {}", exc)
        return {}

    normalized_counts = defaultdict(int)
    for item in data1:
        group = item.get("group", [])
        if len(group) >= 2:
            kw = normalize_topic(group[1])
            normalized_counts[kw] += item.get("current", {}).get("count", 0)

    return dict(normalized_counts)


def extract_topics(html: str) -> list[str]:
    """Extract, rank and canonicalise the research topics from `data1`.

    Topics are ranked by descending count, then near-duplicate variants
    (plurals / inflections) are merged so the list stored in the graph is
    already clean. The frequency-ranked order is preserved by feeding the
    ranked list (most-used first) into the canonicaliser.
    """
    counts = extract_topic_counts(html)
    ranked = sorted(counts, key=lambda topic: counts[topic], reverse=True)
    return canonicalize_topics(ranked)


def build_publications(df_ajax: pl.DataFrame) -> list[Publication]:
    """Build validated Publication models from AJAX dataset.

    :param df_ajax: Cleaned DataFrame from AJAX data.
    :return: List of validated Publication models with capitalized node names.
    """
    if df_ajax.is_empty() or "TIPO" not in df_ajax.columns:
        return []

    df_publications = (
        df_ajax.filter(
            pl.col("TIPO").is_in(
                [
                    "Publicaciones",
                    "Publicaciones Docentes",
                    "Publicaciones de Investigación",
                ]
            )
        )
        .rename(
            {
                "PORTAL_UPM_ID": "portal_upm_id",
                "ACTIVIDAD": "title",
                "AUTORES": "authors",
                "KEYWORDS": "keywords",
                "FECHA INICIO": "start_date",
                "FECHA FIN": "end_date",
                "IDENTIFICADOR": "oa_id",
                "FUENTE": "source",
                "VALIDACIÓN": "validation_agency",
                "SUBTIPO": "type",
                "URL": "url",
                "DESCRIPTION": "description",
            }
        )
        .drop("TIPO")
        .with_columns(
            pl.format("pub_{}", pl.col("portal_upm_id")).alias("id"),
            pl.col("start_date")
            .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
            .dt.year()
            .alias("year"),
            pl.when(
                pl.col("oa_id")
                .cast(pl.String)
                .str.strip_chars()
                .str.contains(r"^https://oa\.upm\.es/[^/\s]+/$")
            )
            .then(pl.col("oa_id").cast(pl.String).str.strip_chars())
            .otherwise(None)
            .alias("oa_id"),
        )
    )

    publications = []
    for record in df_publications.to_dicts():
        title = str(record.get("title") or "").strip()
        if not title:
            logger.warning(
                "Skipping publication with empty title (portal_upm_id={})",
                record.get("portal_upm_id"),
            )
            continue

        raw_url = record.get("url")
        oa_id = record.get("oa_id")
        portal_upm_id = record.get("portal_upm_id")
        record["title"] = title
        description = choose_publication_description(record)

        if raw_url and str(raw_url).strip():
            urls = [u.strip() for u in str(raw_url).split(",,") if u.strip()]
            url_final = urls if len(urls) > 1 else urls[0]
        elif oa_id:
            url_final = oa_id
        else:
            url_final = (
                f"https://portalcientifico.upm.es/es/ipublic/item/{portal_upm_id}"
            )

        # Convert to validated Pydantic model
        pub = Publication(
            id=f"{record['id']}",
            oa_id=oa_id,
            title=title,
            description=description,
            source=record.get("source"),
            validation_agency=record.get("validation_agency"),
            url=url_final,
            year=int(record.get("year", 0)) if record.get("year") else 0,
            type=record.get("type", ""),
            authors=[
                a.strip()
                for a in str(record.get("authors", "")).split(",")
                if a.strip()
            ]
            if record.get("authors")
            else [],
            keywords=canonicalize_topics(
                split_publication_keywords(choose_keyword_source(record))
            ),
            start_date=record.get("start_date"),
            end_date=record.get("end_date"),
        )
        publications.append(pub)

    return publications


def build_profile_document(
    researcher_id: int,
    rf_signature: Optional[str],
    sel: Selector,
    html: str,
    df_ajax: pl.DataFrame,
) -> tuple[
    Researcher,
    list[Publication],
    list[Authorship],
    list[Position],
    list[Works],
    list[Subject],
    list[Teaching],
    list[LastYearProject],
    list[Tutoring],
]:
    """Build validated Pydantic models for a researcher profile and all related data.

    :param researcher_id: The unique ID of the researcher.
    :param rf_signature: The "APELLIDOS, NOMBRE" signature from the CSV.
    :param sel: Scrapy Selector with parsed main HTML.
    :param html: Raw HTML content for topic extraction.
    :param df_ajax: Cleaned DataFrame from AJAX data.
    :return: Tuple of validated Pydantic models: (profile, publications, authorship_edges, positions, works_edges, subjects, teaching_edges, last_year_projects, tutoring_edges).
    """
    full_name, first_name, last_name, email = load_personal_data(sel, rf_signature)
    positions = load_current_positions(sel)
    external_ids = load_external_ids(sel)
    topics = extract_topics(html)
    publications = build_publications(df_ajax)
    subjects = build_subjects(df_ajax)
    last_year_projects = build_last_year_project(df_ajax)

    # Build validated Researcher model with capitalized node name
    profile = Researcher(
        id=f"pdi_{researcher_id}",
        full_name=full_name or "",
        first_name=first_name,
        last_name=last_name,
        email=email,
        external_ids=external_ids,
        topics=topics,
    )

    edges_pdi_publication = build_authorship_edges(
        pdi_id=profile.id, publications=publications
    )
    edges_pdi_positions = build_works_edges(pdi_id=profile.id, positions=positions)
    edges_pdi_subject = build_teaching_edges(pdi_id=profile.id, subjects=subjects)
    edges_pdi_lyp = build_tutoring_edges(
        pdi_id=profile.id, last_year_projects=last_year_projects
    )

    # Remove temporal metadata from position nodes (stored in edges)
    for pos in positions:
        pos.role = None
        pos.since = None

    # Remove temporal metadata from subject nodes (stored in edges)
    for subj in subjects:
        subj.start_date = None
        subj.end_date = None

    return (
        profile,
        publications,
        edges_pdi_publication,
        positions,
        edges_pdi_positions,
        subjects,
        edges_pdi_subject,
        last_year_projects,
        edges_pdi_lyp,
    )


def researcher_exists_in_graph_db(db: GraphDataBase, researcher_id: int) -> bool:
    """Check if a researcher has already been processed in the graph database.

    :param db: The graph database connection object (ArangoDB, Memgraph, etc.).
    :param researcher_id: The ID of the researcher to check.
    :return: True if the researcher exists in the 'pdi' collection/label, False otherwise.
    """
    try:
        pdi_id = f"pdi_{researcher_id}"
        return db.document_exists("pdi", pdi_id)
    except Exception:
        return False


def researcher_exists_locally(researcher_id: int, output_dir: Path) -> bool:
    """Check if a researcher has already been processed locally.

    :param researcher_id: The ID of the researcher to check.
    :param output_dir: The directory where JSON files are saved.
    :return: True if the PDI JSON file exists, False otherwise.
    """
    try:
        matching_files = list(output_dir.glob(f"{researcher_id}_*_pdi.json"))
        return len(matching_files) > 0
    except Exception:
        return False


def researcher_exists_in_minio(researcher_id: int, prefix: str) -> bool:
    """Check if a researcher has already been processed in MinIO.

    :param researcher_id: The ID of the researcher to check.
    :param prefix: The prefix to use for MinIO object keys (unused, kept for API compat).
    :return: True if the PDI object exists in the processed bucket, False otherwise.
    """
    try:
        return minio.object_exists(minio.PROCESSED_BUCKET, f"pdi_{researcher_id}.json")
    except Exception:
        return False


def is_researcher_already_processed(
    researcher_id: int,
    storage: str = "arango",
    db: GraphDataBase = None,
    output_dir: Path = Path("data/processed"),
    minio_prefix: str = "pdi",
) -> bool:
    """Check if a researcher has already been processed in the selected backend.

    :param researcher_id: The ID of the researcher to check.
    :param storage: The storage backend (arango, memgraph, local, or minio).
    :param db: The graph database connection (required for arango/memgraph backends).
    :param output_dir: The output directory (required for local backend).
    :param minio_prefix: The MinIO prefix (required for minio backend).
    :return: True if the researcher has already been processed, False otherwise.
    """
    if storage in ("arango", "memgraph"):
        return researcher_exists_in_graph_db(db, researcher_id)
    elif storage == "local":
        return researcher_exists_locally(researcher_id, output_dir)
    elif storage == "minio":
        return researcher_exists_in_minio(researcher_id, minio_prefix)
    return False


def process_researcher(
    researcher_id: int,
    rf_signature: Optional[str],
    storage: str = "arango",
    db: Optional[GraphDataBase] = None,
    output_dir: Path = Path("data/processed"),
    minio_prefix: str = "pdi",
    raw_dir: Path = Path("data/raw"),
    skip_existing: bool = False,
    html_source: str = "local",
    html_minio_prefix: str = "raw",
) -> dict:
    """Run the full local processing pipeline for one researcher.

    :param researcher_id: The unique ID of the researcher to process.
    :param rf_signature: The "APELLIDOS, NOMBRE" signature from the CSV for this researcher, used to extract names.
    :param storage: The storage backend to use for saving results. Can be 'arango', 'local', or 'minio'.
    :param db: An optional database connection object, required if storage is 'arango'.
    :param output_dir: The directory where the JSON files will be saved.
    :param minio_prefix: The prefix for the Minio bucket (for output storage).
    :param raw_dir: The directory containing the raw HTML files (used if html_source='local').
    :param skip_existing: If True, skip processing if the researcher has already been processed in the selected backend.
    :param html_source: Source to read HTML files from ('local' or 'minio'). Defaults to 'local'.
    :param html_minio_prefix: MinIO prefix for reading HTML files (used if html_source='minio'). Defaults to 'raw'.
    :return: A dictionary containing the processing status and result data.
    """
    try:
        # Check if researcher has already been processed
        if skip_existing and is_researcher_already_processed(
            researcher_id, storage, db, output_dir, minio_prefix
        ):
            logger.debug("Skipping {}: already processed", researcher_id)
            return {"status": "skipped_existing", "researcher_id": researcher_id}

        # Load HTML files from local or MinIO
        main_html = load_html_file(
            researcher_id,
            "main",
            html_source=html_source,
            raw_dir=raw_dir,
            minio_prefix=html_minio_prefix,
        )
        ajax_html = load_html_file(
            researcher_id,
            "ajax",
            html_source=html_source,
            raw_dir=raw_dir,
            minio_prefix=html_minio_prefix,
        )

        if main_html is None or ajax_html is None:
            missing = []
            if main_html is None:
                missing.append(f"{researcher_id}_main.html ({html_source})")
            if ajax_html is None:
                missing.append(f"{researcher_id}_ajax.html ({html_source})")
            logger.warning(
                "Skipping {}: missing files -> {}", researcher_id, ", ".join(missing)
            )
            return {
                "status": "missing_files",
                "researcher_id": researcher_id,
                "missing": missing,
            }

        html = main_html
        sel = Selector(text=html)

        raw_activity = _parse_activity_json(ajax_html)
        if raw_activity is not None:
            minio.upload_json(
                minio.PROCESSED_BUCKET,
                raw_activity,
                f"pdi_{researcher_id}.json",
            )

        df_ajax = load_ajax_dataSet(ajax_html)

        (
            profile,
            publications,
            authorship_edges,
            positions,
            works_edges,
            subjects,
            teaching_edges,
            last_year_projects,
            tutoring_edges,
        ) = build_profile_document(
            researcher_id=researcher_id,
            rf_signature=rf_signature,
            sel=sel,
            html=html,
            df_ajax=df_ajax,
        )

        # Register only COUNTS in memory (keeping the full objects here for the
        # whole dataset causes unbounded RAM growth -> OOM / exit code 137).
        PUBLICATIONS_REGISTRY[researcher_id] = len(publications)
        SUBJECTS_REGISTRY[researcher_id] = len(subjects)
        LYP_REGISTRY[researcher_id] = len(last_year_projects)

        # Save according to storage backend
        success = False
        output_path = None

        if storage in ("arango", "memgraph"):
            if db is None:
                raise ValueError(
                    f"Graph database connection not available for {storage}"
                )
            success = save_to_graph_db(
                db,
                profile,
                publications,
                positions,
                works_edges,
                subjects,
                authorship_edges,
                teaching_edges,
                last_year_projects,
                tutoring_edges,
            )
        elif storage == "local":
            success = save_to_local(
                profile, publications, last_year_projects, researcher_id, output_dir
            )
            if success:
                safe_last_name = normalize_last_name(profile.last_name)
                output_path = output_dir / f"{researcher_id}_{safe_last_name}_pdi.json"
        elif storage == "minio":
            success = save_to_minio(
                profile, publications, last_year_projects, researcher_id, minio_prefix
            )
        else:
            raise ValueError(f"Unknown storage backend: {storage}")

        if not success:
            logger.warning(
                "PDI {} could not be saved (storage={})", researcher_id, storage
            )
            return {
                "status": "failed",
                "researcher_id": researcher_id,
                "error": f"Failed to save to {storage}",
            }

        logger.info(
            "PDI {} processed: publications={}, subjects={}, last_year_projects={}",
            researcher_id,
            len(publications),
            len(subjects),
            len(last_year_projects),
        )

        return {
            "status": "processed",
            "researcher_id": researcher_id,
            "profile": profile,
            "publications": publications,
            "subjects": subjects,
            "last_year_projects": last_year_projects,
            "authorship_edges": authorship_edges
            if storage in ("arango", "memgraph")
            else None,
            "teaching_edges": teaching_edges
            if storage in ("arango", "memgraph")
            else None,
            "tutoring_edges": tutoring_edges
            if storage in ("arango", "memgraph")
            else None,
            "output_path": output_path,
        }
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        logger.exception("PDI {} failed: {}", researcher_id, exc)
        return {
            "status": "failed",
            "researcher_id": researcher_id,
            "error": str(exc),
        }


def process_all_researchers(
    researcher_rows: list[tuple[int, str | None]],
    researcher_id: Optional[int] = None,
    limit: Optional[int] = None,
    storage: str = "arango",
    db=None,
    output_dir: Path = Path("data/processed"),
    minio_prefix: str = "pdi",
    skip_existing: bool = False,
    progress_every: int = 100,
    html_source: str = "local",
    html_minio_prefix: str = "raw",
    raw_dir: Path = Path("data/raw"),
) -> list[dict]:
    """Process one researcher or the full local dataset.
    It accepts the researcher rows (ID and RF signature) as input, along with optional parameters to filter by a specific researcher ID or limit the number of researchers processed.
    The function also takes parameters to specify the storage backend (ArangoDB, local JSON files, or MinIO) and the necessary configurations for each storage type.
    It returns a list of dictionaries containing the results of processing each researcher, including their profile, publications, edges (if applicable), and output paths.

    :param researcher_rows: A list of tuples containing researcher IDs and their corresponding RF signatures, typically loaded from the CSV file.
    :param researcher_id: Optional specific researcher ID to process. If None, all researchers in the list will be processed.
    :param limit: Optional limit on the number of researchers to process. If None, all researchers (or the one specified by researcher_id) will be processed.
    :param storage: The storage backend to use for saving results. Can be 'arango', 'memgraph', 'local', or 'minio'.
    :param db: An optional graph database connection object, required if storage is 'arango' or 'memgraph'.
    :param output_dir: The directory to save local JSON files if storage is 'local'.
    :param minio_prefix: The prefix to use for MinIO object keys if storage is 'minio'.
    :param skip_existing: If True, skip researchers that have already been processed in the selected backend.
    :param progress_every: Emit a progress log every N researchers in batch mode. Use 0 to disable.
    :param html_source: Source to read HTML files from ('local' or 'minio'). Defaults to 'local'.
    :param html_minio_prefix: MinIO prefix for reading HTML files (used if html_source='minio'). Defaults to 'raw'.
    :param raw_dir: The directory containing the raw HTML files (used if html_source='local').
    :return: A list of dictionaries containing the results of processing each researcher, including their profile, publications, edges (if applicable), and output paths.
    """
    rows = researcher_rows
    if researcher_id is not None:
        rows = [row for row in rows if row[0] == researcher_id]
    if limit is not None:
        rows = rows[:limit]

    total_rows = len(rows)
    if total_rows == 0:
        logger.warning("No researchers selected for processing")
        return []

    if total_rows > 1:
        logger.info(
            "Starting batch processing: total={}, storage={}, html_source={}, skip_existing={}",
            total_rows,
            storage,
            html_source,
            skip_existing,
        )

    batch_start = time.perf_counter()

    results = []
    stats = {"processed": 0, "skipped_existing": 0, "missing_files": 0, "failed": 0}
    for idx, (current_id, rf_signature) in enumerate(rows, start=1):
        try:
            result = process_researcher(
                current_id,
                rf_signature,
                storage=storage,
                db=db,
                output_dir=output_dir,
                minio_prefix=minio_prefix,
                skip_existing=skip_existing,
                html_source=html_source,
                html_minio_prefix=html_minio_prefix,
                raw_dir=raw_dir,
            )
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.exception("Unexpected batch error for PDI {}: {}", current_id, exc)
            result = {
                "status": "failed",
                "researcher_id": current_id,
                "error": str(exc),
            }

        status = result.get("status", "failed")
        stats[status] = stats.get(status, 0) + 1

        if status == "processed":
            # Keep only a lightweight record.
            # Appending the full result (profile, publications, subjects, edges...) retains the entire dataset in RAM and leads to OOM (exit code 137) on large batches.
            results.append(
                {
                    "status": status,
                    "researcher_id": result.get("researcher_id"),
                }
            )

        should_emit_progress = (
            total_rows > 1
            and progress_every > 0
            and (idx % progress_every == 0 or idx == total_rows)
        )
        if should_emit_progress:
            elapsed_seconds = time.perf_counter() - batch_start
            throughput = (idx / elapsed_seconds) if elapsed_seconds > 0 else 0.0
            remaining_items = total_rows - idx
            eta_seconds = (remaining_items / throughput) if throughput > 0 else 0.0

            logger.info(
                "Batch progress: {}/{} | processed={} skipped={} missing={} failed={} | elapsed={} eta={}",
                idx,
                total_rows,
                stats["processed"],
                stats["skipped_existing"],
                stats["missing_files"],
                stats["failed"],
                format_duration(elapsed_seconds),
                format_duration(eta_seconds),
            )

    total_elapsed = time.perf_counter() - batch_start
    logger.info(
        "Execution summary: processed={}, skipped_existing={}, missing_files={}, failed={}, elapsed={}",
        stats["processed"],
        stats["skipped_existing"],
        stats["missing_files"],
        stats["failed"],
        format_duration(total_elapsed),
    )

    return results


def save_to_graph_db(
    db: GraphDataBase,
    profile: Researcher,
    publications: list[Publication],
    positions: list[Position],
    works_edges: list[Works],
    subjects: list[Subject],
    authorship_edges: list[Authorship],
    teaching_edges: list[Teaching],
    last_year_projects: list[LastYearProject],
    tutoring_edges: list[Tutoring],
) -> bool:
    """Save validated Pydantic models to graph database, converting to backend-specific formats.

    :param db: The graph database connection object implementing GraphDataBase interface.
    :param profile: The Researcher instance to save.
    :param publications: A list of Publication instances to save.
    :param positions: A list of Position instances to save.
    :param works_edges: A list of Works edge instances to save.
    :param subjects: A list of Subject instances to save.
    :param authorship_edges: A list of Authorship edge instances to save.
    :param teaching_edges: A list of Teaching edge instances to save.
    :param last_year_projects: A list of LastYearProject instances to save.
    :param tutoring_edges: A list of Tutoring edge instances to save.
    :return: True if the save operation was successful, False otherwise.
    """
    try:
        db.ensure_collections()

        # Convert Pydantic models to dicts
        profile_dict = profile.model_dump()
        publications_dicts = [pub.model_dump() for pub in publications]
        authorship_dicts = [edge.model_dump() for edge in authorship_edges]
        positions_dicts = [pos.model_dump() for pos in positions]
        works_dicts = [edge.model_dump() for edge in works_edges]
        subjects_dicts = [subj.model_dump() for subj in subjects]
        teaching_dicts = [edge.model_dump() for edge in teaching_edges]
        last_year_projects_dicts = [lyp.model_dump() for lyp in last_year_projects]
        tutoring_edge_dicts = [edge.model_dump() for edge in tutoring_edges]

        # Apply backend-specific adaptations if backend supports adapt_document
        if hasattr(db, "adapt_document"):
            profile_dict = db.adapt_document("Pdi", profile_dict)
            publications_dicts = [
                db.adapt_document("Publication", d) for d in publications_dicts
            ]
            positions_dicts = [
                db.adapt_document("Position", d) for d in positions_dicts
            ]
            works_dicts = [db.adapt_document("Works", d) for d in works_dicts]
            subjects_dicts = [db.adapt_document("Subject", d) for d in subjects_dicts]
            last_year_projects_dicts = [
                db.adapt_document("LastYearProject", d)
                for d in last_year_projects_dicts
            ]
            authorship_dicts = [
                db.adapt_document("Authorship", d) for d in authorship_dicts
            ]
            teaching_dicts = [db.adapt_document("Teaching", d) for d in teaching_dicts]
            tutoring_edge_dicts = [
                db.adapt_document("Tutoring", d) for d in tutoring_edge_dicts
            ]

        # Upsert documents with proper collection names
        db.upsert_documents("Pdi", [profile_dict])
        db.upsert_documents("Publication", publications_dicts)
        db.upsert_edges("Authorship", authorship_dicts)
        db.upsert_documents("Position", positions_dicts)
        db.upsert_edges("Works", works_dicts)
        db.upsert_documents("Subject", subjects_dicts)
        db.upsert_edges("Teaching", teaching_dicts)
        db.upsert_documents("LastYearProject", last_year_projects_dicts)
        db.upsert_edges("Tutoring", tutoring_edge_dicts)

        logger.debug(
            "Saved to graph DB: pdi={}, pubs={}, positions={}, subjects={}, last_year_projects={}, authorship_edges={}, works_edges={}, teaching_edges={}, tutoring_edges={}",
            profile.id,
            len(publications),
            len(positions),
            len(subjects),
            len(last_year_projects),
            len(authorship_edges),
            len(works_edges),
            len(teaching_edges),
            len(tutoring_edges),
        )
        return True
    except Exception as exc:
        logger.exception(
            "Error saving to graph database for {}: {}",
            profile.id if profile else "unknown",
            exc,
        )
        return False


def save_to_local(
    profile: Researcher,
    publications: list[Publication],
    last_year_projects: list[LastYearProject],
    researcher_id: int,
    output_dir: Path = Path("data/processed"),
) -> bool:
    """Save validated Pydantic models to local JSON files (no edges).

    :param profile: The Researcher instance to save.
    :param publications: A list of Publication instances to save.
    :param last_year_projects: A list of LastYearProject instances to save.
    :param researcher_id: The unique ID of the researcher, used for naming the output files.
    :param output_dir: The directory where the JSON files will be saved.
    :return: True if the save operation was successful, False otherwise.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_last_name = normalize_last_name(profile.last_name, fallback="N_A")

        profile_path = output_dir / f"{researcher_id}_{safe_last_name}_pdi.json"
        profile_path.write_text(
            json.dumps(profile.model_dump(), ensure_ascii=False, indent=4),
            encoding="utf-8",
        )

        if publications:
            pub_path = (
                output_dir / f"{researcher_id}_{safe_last_name}_publications.json"
            )
            pub_path.write_text(
                json.dumps(
                    [pub.model_dump() for pub in publications],
                    ensure_ascii=False,
                    indent=4,
                ),
                encoding="utf-8",
            )

        if last_year_projects:
            last_year_projects_path = (
                output_dir / f"{researcher_id}_{safe_last_name}_last_year_projects.json"
            )
            last_year_projects_path.write_text(
                json.dumps(
                    [
                        LastYearProject.model_dump()
                        for LastYearProject in last_year_projects
                    ],
                    ensure_ascii=False,
                    indent=4,
                ),
                encoding="utf-8",
            )

        logger.debug(
            "Saved locally: {}, pubs={}, last_year_projects={}",
            profile_path,
            len(publications),
            len(last_year_projects),
        )
        return True
    except Exception as exc:
        logger.exception("Error saving locally for {}: {}", researcher_id, exc)
        return False


def save_to_minio(
    profile: Researcher,
    publications: list[Publication],
    last_year_projects: list[LastYearProject],
    researcher_id: int,
    prefix: str = "pdi",
) -> bool:
    """Save validated Pydantic models to MinIO (no edges).
    Uses a temporary directory to create the JSON files before uploading them to MinIO, and then cleans up the temporary files automatically.

    :param profile: The Researcher instance to save.
    :param publications: A list of Publication instances to save.
    :param last_year_projects: A list of LastYearProject instances to save.
    :param researcher_id: The unique ID of the researcher, used for naming the MinIO objects.
    :param prefix: The prefix to use for the MinIO object keys (default is "pdi").
    :return: True if the save operation was successful, False otherwise.
    """
    try:
        safe_last_name = normalize_last_name(profile.last_name)

        pdi_key = f"{prefix}/{researcher_id}_{safe_last_name}_pdi.json"
        if not minio.upload_json(minio.PROCESSED_BUCKET, profile.model_dump(), pdi_key):
            return False

        if publications:
            publications_key = (
                f"{prefix}/{researcher_id}_{safe_last_name}_publications.json"
            )
            if not minio.upload_json(
                minio.PROCESSED_BUCKET,
                [pub.model_dump() for pub in publications],
                publications_key,
            ):
                return False

        if last_year_projects:
            last_year_projects_key = (
                f"{prefix}/{researcher_id}_{safe_last_name}_last_year_projects.json"
            )
            if not minio.upload_json(
                minio.PROCESSED_BUCKET,
                [
                    LastYearProject.model_dump()
                    for LastYearProject in last_year_projects
                ],
                last_year_projects_key,
            ):
                return False

        logger.debug(
            "Saved to MinIO: {}/{}_{}_* , pubs={}, last_year_projects={}",
            prefix,
            researcher_id,
            safe_last_name,
            len(publications),
            len(last_year_projects),
        )
        return True
    except Exception as exc:
        logger.exception("Error saving to MinIO for {}: {}", researcher_id, exc)
        return False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the local processing pipeline.

    :return: An argparse.Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process UPM PDI profiles from local HTML files."
    )
    parser.add_argument(
        "-i", "--id", type=int, default=None, help="Process only one researcher ID."
    )
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=None,
        help="Limit the number of researchers processed.",
    )
    parser.add_argument(
        "-s",
        "--storage",
        type=str,
        default="memgraph",
        choices=["arango", "memgraph", "local", "minio"],
        help="Storage backend: arango (default), memgraph, local (JSON files), or minio.",
    )
    parser.add_argument(
        "-o",
        "--local-out-dir",
        type=str,
        default="data/processed",
        help="Output directory for local storage (used only if --storage=local).",
    )
    parser.add_argument(
        "-m",
        "--minio-prefix",
        type=str,
        default="pdi",
        help="Prefix for MinIO objects (used only if --storage=minio).",
    )
    parser.add_argument(
        "-k",
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip researchers that have already been processed in the selected backend.",
    )
    parser.add_argument(
        "--logs",
        type=int,
        default=20,
        help="Numeric log level for loguru (10=DEBUG,20=INFO,30=WARNING,40=ERROR,50=CRITICAL). Default is 20 (INFO).",
    )
    parser.add_argument(
        "-p",
        "--progress-every",
        type=int,
        default=100,
        help=(
            "Progress log frequency for batch mode (every N researchers). "
            "Use 0 to disable periodic progress logs."
        ),
    )
    parser.add_argument(
        "--html-source",
        type=str,
        default="minio",
        choices=["local", "minio"],
        help="Source to read HTML files from: local reads from `data/raw/`, minio (default) reads from MinIO bucket.",
    )
    parser.add_argument(
        "--html-minio-prefix",
        type=str,
        default="researchers",
        help="MinIO prefix for reading HTML files (used only if --html-source=minio). Default is 'researchers'.",
    )
    return parser.parse_args()


def _build_derived_relationships(db: GraphDataBase) -> None:
    """Materialise derived graph relationships after all PDI data has been loaded.

    Runs only when the storage backend is Memgraph. Safe to re-run: MERGE is
    idempotent so existing edges are updated in place rather than duplicated.
    """
    logger.info("Building derived graph relationships...")

    coauthored_query = """
    MATCH (a:Pdi)-[:Authorship]->(p:Publication)<-[:Authorship]-(b:Pdi)
    WHERE a.id < b.id
    WITH a, b,
         count(p)    AS num,
         min(p.year) AS first_year,
         max(p.year) AS last_year
    MERGE (a)-[r:Coauthored]-(b)
    SET r.num = num, r.first_year = first_year, r.last_year = last_year
    """
    db.query(coauthored_query)
    logger.info("Coauthored relationships built")

    # department_query = """
    # MATCH (a:Pdi)-[:Works]->(pos:Position)<-[:Works]-(b:Pdi)
    # WHERE a.id < b.id
    # WITH a, b,
    #      count(DISTINCT pos)          AS num,
    #      collect(DISTINCT pos.name)   AS depts,
    #      collect(DISTINCT pos.entity) AS entity_types
    # MERGE (a)-[r:SharesDept]-(b)
    # SET r.num = num, r.depts = depts, r.entity_types = entity_types
    # """
    # db.query(department_query)
    # logger.info("Department relationships built")


def final_summary(args, results: list[dict], researchers: list[tuple]):
    """Print final summary and statistics regardless of how the program exits.

    :param args: Parsed command-line arguments.
    :param results: List of processing results.
    :param researchers: List of researcher tuples (id, name).
    """
    try:
        logger.info(
            "Pipeline finished: processed={}, available={}, storage={}, html_source={}",
            len(results),
            len(researchers),
            args.storage,
            args.html_source,
        )

        if args.storage == "local":
            logger.info("Output directory: {}", args.local_out_dir)
        elif args.storage == "minio":
            logger.info("MinIO prefix: {}", args.minio_prefix)

        if args.html_source == "minio":
            logger.info("HTML source: MinIO prefix='{}'", args.html_minio_prefix)
        else:
            logger.info("HTML source: local directory (data/raw/)")

        total_pubs = sum(PUBLICATIONS_REGISTRY.values())
        total_subjects = sum(SUBJECTS_REGISTRY.values())
        total_lyp = sum(LYP_REGISTRY.values())
        logger.info(
            "Final summary: {} researchers, {} total publications, {} total subjects, {} total last year projects",
            len(PUBLICATIONS_REGISTRY),
            total_pubs,
            total_subjects,
            total_lyp,
        )
    except Exception as exc:
        logger.error("Error printing final summary: {}", exc)


if __name__ == "__main__":
    args = parse_args()

    setup_logging("create_pdis", level=args.logs)
    minio.ensure_bucket(minio.PROCESSED_BUCKET)

    results = []
    researchers = []
    arango_db = None

    try:
        # Initialize storage backend
        if args.storage in ("arango", "memgraph"):
            arango_db = get_graph_db(args.storage)
            logger.info("Connected to {} database", args.storage.upper())

        ids, names = load_researcher_ids_with_signatures()
        researchers = list(zip(ids, names))

        results = process_all_researchers(
            researcher_rows=researchers,
            researcher_id=args.id,
            limit=args.limit,
            storage=args.storage,
            db=arango_db,
            output_dir=Path(args.local_out_dir),
            minio_prefix=args.minio_prefix,
            skip_existing=args.skip_existing,
            progress_every=args.progress_every,
            html_source=args.html_source,
            html_minio_prefix=args.html_minio_prefix,
            raw_dir=Path("data/raw"),
        )

        if args.storage == "memgraph" and arango_db is not None:
            _build_derived_relationships(arango_db)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
    except Exception as exc:
        logger.error("Pipeline failed with error: {}", exc, exc_info=True)
    finally:
        final_summary(args, results, researchers)
