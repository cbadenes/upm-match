"""Text normalisation and cleaning utilities shared across the pipeline."""

import re
from unicodedata import normalize as unicodedata_normalize

# Markers used by clean_description

PUBLICATION_TRAILING_MARKERS = [
    "(Summary)",
    "(Notes)",
    "(Objectives)",
    "(Most relevant results)",
    "(Degree of originality and innovation)",
    "(Methodological contribution)",
    "(Thematic priority)",
    "(Use of the contribution)",
    "(Cites received for the contribution)",
    "(Quality of the publication/dissemination/exhibition medium)",
    "(Non-academic mentions and references to the contribution)",
    "(Copyright held by the owner/author(s))",
]

PUBLICATION_PREFIX_REGEXES = [
    re.compile(r"^\s*Abstract:\s*", re.IGNORECASE),
    re.compile(r"^\s*©\s*\d{4}\s*(?:Copyright\s+)?", re.IGNORECASE),
]

PUBLICATION_BILINGUAL_ABSTRACT_REGEX = re.compile(
    r"\s*-{2,}\s*ABSTRACT\s*-{2,}\s*", re.IGNORECASE
)

PUBLICATION_ABSTRACT_SECTION_REGEX = re.compile(
    r"\b(?:BACKGROUND|OBJECTIVES?|AIM|AIMS|METHODS?|RESULTS?|CONCLUSIONS?|INTRODUCTION|PURPOSES?)\s*:",
    re.IGNORECASE,
)

PUBLICATION_AUTHOR_BLOCK_MARKERS = [
    "(Authors with Filk)",
    "(Author of contact)",
    "(Recognitions linked to item)",
]

PUBLICATION_AUTHOR_START_REGEX = re.compile(
    r"^\s*[A-ZÁÉÍÓÚÜÑ][^.;]{0,80}(?:,\s*[A-ZÁÉÍÓÚÜÑ][^;]{0,10})?",
    re.IGNORECASE,
)

PUBLICATION_AUTHOR_CONTEXT_REGEX = re.compile(
    r"\b(University|Universidad|Department|Departamento|Institute|Centro|School|College|Research|Spain|Ltd|London|Madrid)\b",
    re.IGNORECASE,
)

PUBLICATION_COPYRIGHT_REGEX = re.compile(
    r"(?:©\s*\d{4}\s*)?(?:Copyright\s+for this paper by its authors\.|Copyright\s+held by the owner/author\(s\)\.)",
    re.IGNORECASE,
)

PUBLICATION_MISSING_STATEMENT_REGEX = re.compile(
    r"^\s*No Statement Available(?:\s*\(Recognitions linked to item\))?\s*$",
    re.IGNORECASE,
)

PUBLICATION_METADATA_REGEX = re.compile(
    r"\bDOI:\s*\S+|\b\d{4};\s*\d+\b|\b\d{4}\s*\(\d+\):\s*-|\b\d{4}\s*\(\):\s*-",
    re.IGNORECASE,
)


# Topic / keyword normalisation


def normalize_topic(topic: str) -> str:
    """Normalise a research topic or keyword string.

    Steps: NFKD → ASCII, lowercase, strip, remove trailing period, collapse
    simple plurals (e.g. "graphs" → "graph" but not "process"), collapse spaces.
    """
    t = unicodedata_normalize("NFKD", topic)
    t = t.encode("ascii", "ignore").decode("ascii")
    t = t.lower().strip()
    t = t.rstrip(".")
    t = re.sub(r"(?<=[a-z]{4})s\b", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def split_publication_keywords(value: object) -> list[str]:
    """Split and normalise publication keywords while preserving order."""
    if value is None:
        return []

    raw_values: list = (
        value if isinstance(value, list) else re.split(r"[;|/]", str(value))
    )

    keywords: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        keyword = (
            normalize_topic(
                re.sub(r"\s+", " ", str(raw_value).replace("\xa0", " ")).strip()
            )
            if raw_value
            else ""
        )
        if not keyword or keyword in seen:
            continue
        seen.add(keyword)
        keywords.append(keyword)

    return keywords


def has_keywords(value: object) -> bool:
    """Return True when a keyword-like value contains at least one normalised token."""
    return bool(split_publication_keywords(value))


def choose_keyword_source(record: dict[str, object]) -> object | None:
    """Pick the keyword field, falling back to categories when keywords are missing."""
    primary_keywords = record.get("keywords")
    if has_keywords(primary_keywords):
        return primary_keywords

    fallback_keywords = record.get("CATEGORÍAS")
    if has_keywords(fallback_keywords):
        return fallback_keywords

    return None


# Abstract / description cleaning


def clean_description(value: str | None) -> str | None:
    """Clean and normalise a publication description.

    Removes common prefixes, author blocks, trailing metadata markers,
    bilingual-abstract duplicates, copyright notices, and normalises whitespace.
    Returns None when the result carries no real content.
    """
    if not value:
        return None

    text = value.replace("\xa0", " ").strip()

    if PUBLICATION_MISSING_STATEMENT_REGEX.match(text):
        return None

    for prefix_regex in PUBLICATION_PREFIX_REGEXES:
        text = prefix_regex.sub("", text)

    bilingual_parts = PUBLICATION_BILINGUAL_ABSTRACT_REGEX.split(text, maxsplit=1)
    if len(bilingual_parts) > 1:
        text = bilingual_parts[0].strip()

    if text.startswith(("©", "Copyright")):
        section_match = PUBLICATION_ABSTRACT_SECTION_REGEX.search(text)
        if section_match:
            text = text[section_match.start() :].strip()

    if PUBLICATION_AUTHOR_START_REGEX.match(
        text
    ) and PUBLICATION_AUTHOR_CONTEXT_REGEX.search(text[:250]):
        for marker in PUBLICATION_AUTHOR_BLOCK_MARKERS:
            marker_index = text.find(marker)
            if marker_index != -1:
                text = text[marker_index + len(marker) :]
                text = re.sub(r"^\s*;\s*", "", text)
                break

    match = PUBLICATION_COPYRIGHT_REGEX.search(text)
    if match:
        prefix = text[: match.start()].strip()
        if prefix:
            text = prefix

    noise_index = None
    for marker in PUBLICATION_TRAILING_MARKERS:
        marker_index = text.find(marker)
        if marker_index != -1 and (noise_index is None or marker_index < noise_index):
            noise_index = marker_index

    if noise_index is not None:
        text = text[:noise_index]

    if re.search(r"\b(?:DOI|arXiv)\b", text, re.IGNORECASE):
        citation_tail_match = re.search(r"\s+\d{4}\s*[;,].*$|\s*,\s*\(\):\s*.*$", text)
        if citation_tail_match:
            prefix = text[: citation_tail_match.start()].strip()
            if prefix:
                text = prefix

    metadata_match = PUBLICATION_METADATA_REGEX.search(text)
    if metadata_match:
        prefix = text[: metadata_match.start()].strip()
        if prefix:
            text = prefix

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([,.;:])\s*$", r"\1", text).strip()
    text = re.sub(r"[\s,;:.]+$", "", text).strip()

    content_tokens = [
        token
        for token in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", text)
        if token.lower() not in {"doi", "arxiv"}
    ]
    if not content_tokens:
        return None

    return text or None


def choose_publication_description(record: dict[str, object]) -> str:
    """Extract the most relevant description for a publication from a raw record.

    Prefers the longest of 'description' / 'DESCRIPCION TFT', cleans it, and
    falls back to the raw text when cleaning strips everything meaningful.
    Returns an empty string when no usable content is found.
    """
    raw_description = record.get("description")
    other_description = record.get("DESCRIPCION TFT")

    raw_description = (
        other_description
        if len(str(other_description)) > len(str(raw_description))
        else raw_description
    )
    if raw_description is None:
        return ""

    cleaned_description = clean_description(raw_description)
    if cleaned_description:
        return cleaned_description

    raw_text = re.sub(r"\s+", " ", str(raw_description).replace("\xa0", " ")).strip()
    raw_tokens = [
        token
        for token in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]+", raw_text)
        if token.lower() not in {"doi", "arxiv"}
    ]
    if not raw_tokens:
        return ""

    return raw_text
