"""Text normalisation and cleaning utilities shared across the pipeline."""

import re
from collections import Counter, defaultdict
from unicodedata import normalize as unicodedata_normalize

# Labelled sections used by clean_description
#
# The UPM portal serialises a publication record as a sequence of sections, each section TERMINATED by a parenthesised label and separated by ", ", e.g.:
#
#   <recognitions> (Recognitions linked to item), <contact> (Author of contact),
#   <authors> (Authors with Filk), <abstract> (Summary)
#
# The section ORDER varies between records, so the abstract cannot be assumed to come first: it is the section explicitly labelled "(Summary)" (or one of the content fallbacks below).
# The label vocabulary was inventoried over the full raw corpus (66k publications, 2026-07); it is a closed English set.

# Content labels, in preference order. "(Objectives)" and "(Most relevant results)" belong to the same fallback tier and are joined together.
PUBLICATION_SUMMARY_LABEL = "(Summary)"
PUBLICATION_DESCRIPTION_LABEL = "(General Description)"
PUBLICATION_OBJECTIVES_LABELS = ["(Objectives)", "(Most relevant results)"]

# Metadata labels: their sections never contain the abstract (authors, contact addresses, acknowledgments/funding, CV-merit prose, quality indicators...).
PUBLICATION_METADATA_LABELS = [
    "(Author of contact)",
    "(Authors with Filk)",
    "(Recognitions linked to item)",
    "(Methodological contribution)",
    "(Degree of originality and innovation)",
    "(Thematic priority)",
    "(Use of the contribution)",
    "(Quality of the publication/dissemination/exhibition medium)",
    "(Cites received for the contribution)",
    "(Degree of opening)",
    "(Notes)",
    "(Non-academic mentions and references to the contribution)",
    "(Quality indications)",
    "(Miscellaneous)",
    "(Copyright held by the owner/author(s))",
    # Inline author annotations; treating them as boundaries only splits author/contact blocks further, which is harmless.
    "(corresponding author)",
    "(autor correspondiente)",
    "(reprint author)",
]

PUBLICATION_SECTION_LABELS = (
    [PUBLICATION_SUMMARY_LABEL, PUBLICATION_DESCRIPTION_LABEL]
    + PUBLICATION_OBJECTIVES_LABELS
    + PUBLICATION_METADATA_LABELS
)

# Longest-first so overlapping labels can never shadow each other.
_SECTION_LABEL_REGEX = re.compile(
    "|".join(
        re.escape(label)
        for label in sorted(PUBLICATION_SECTION_LABELS, key=len, reverse=True)
    )
)

# Separator between a section label and the next section ("," in the AJAX
# dataset, ";" in other portal serialisations).
_SECTION_SEPARATOR_REGEX = re.compile(r"^\s*[;,.]\s*")

PUBLICATION_PREFIX_REGEXES = [
    re.compile(r"^\s*Abstract:\s*", re.IGNORECASE),
    re.compile(r"^\s*©\s*\d{4}\s*(?:Copyright\s+)?", re.IGNORECASE),
    # Publisher copyright boilerplate ending in "rights reserved", e.g. "© Springer India 2016. All rights reserved. ..." or "Copyright © 2020 by SCITEPRESS ... Lda. All rights reserved ...".
    # A real abstract never *opens* with that phrase, so a prefix containing it within the first ~100 characters is always boilerplate.
    re.compile(r"^.{0,100}?\brights reserved\b\.?[:\s]*", re.IGNORECASE | re.DOTALL),
]

PUBLICATION_BILINGUAL_ABSTRACT_REGEX = re.compile(
    r"\s*-{2,}\s*ABSTRACT\s*-{2,}\s*", re.IGNORECASE
)

PUBLICATION_ABSTRACT_SECTION_REGEX = re.compile(
    r"\b(?:BACKGROUND|OBJECTIVES?|AIM|AIMS|METHODS?|RESULTS?|CONCLUSIONS?|INTRODUCTION|PURPOSES?)\s*:",
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
    """Normalise a single research topic or keyword string.

    Steps: lowercase, strip a trailing possessive ("'s"), NFKD → ASCII, trim, remove trailing period, collapse spaces.
    Plural/inflection merging is NOT done here (a per-string view cannot tell "analysis" from a real plural), use :func:`canonicalize_topics` on the whole list for that.
    """
    t = topic.lower().strip()
    t = re.sub(r"[''’]s\b", "", t)  # drop possessive ("ontology's" -> "ontology")
    t = unicodedata_normalize("NFKD", t)
    t = t.encode("ascii", "ignore").decode("ascii")
    t = t.strip().rstrip(".").strip()
    t = re.sub(r"\s+", " ", t)
    return t


# Topic / keyword list canonicalisation (merge near-duplicate variants)
_TRIGRAM_FUZZY_THRESHOLD = 0.85  # conservative: only near-identical surface forms


def _char_trigrams(text: str) -> set[str]:
    """Return the set of character trigrams of a compact (spaceless) string."""
    compact = text.replace(" ", "")
    if len(compact) < 3:
        return {compact} if compact else set()
    return {compact[i : i + 3] for i in range(len(compact) - 2)}


def _trigram_jaccard(a: str, b: str) -> float:
    """Jaccard similarity over character trigrams of two strings."""
    ta, tb = _char_trigrams(a), _char_trigrams(b)
    if not ta or not tb:
        return 1.0 if a == b else 0.0
    inter = len(ta & tb)
    return inter / (len(ta) + len(tb) - inter)


def _same_token_stem(a: str, b: str) -> bool:
    """True when two single tokens are the same word up to a plural inflection.

    Handles -s, -es and -y/-ies. This is enough to merge "graph"/"graphs", "disease"/"diseases", "box"/"boxes", "ontology"/"ontologies" and even "coronavirus"/"coronaviruses" (singular already ends in 's').
    A short fuzzy fallback catches light typos / diacritic leftovers.
    """
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if not short:
        return False
    if long in (short + "s", short + "es"):
        return True
    if short.endswith("y") and long == short[:-1] + "ies":
        return True
    # Fuzzy fallback only for clearly similar tokens sharing the first letter.
    if short[0] == long[0] and len(short) >= 4:
        return _trigram_jaccard(a, b) >= _TRIGRAM_FUZZY_THRESHOLD
    return False


def _is_variant(a: str, b: str) -> bool:
    """True when two normalised topics are inflectional variants of each other.

    Compared token by token: they must have the same number of tokens, every token must match or be a plural/inflection of its counterpart, and at least one token must differ (otherwise they are identical).
    The equal-token-count rule keeps distinct concepts apart ("language model" vs "large language model", "topic model" vs "dynamic topic model").
    """
    ta, tb = a.split(), b.split()
    if len(ta) != len(tb) or not ta:
        return False
    differing = 0
    for x, y in zip(ta, tb):
        if x == y:
            continue
        if _same_token_stem(x, y):
            differing += 1
        else:
            return False
    return differing > 0


def canonicalize_topics(topics: list[str] | None) -> list[str]:
    """Normalise and merge near-duplicate topics/keywords into a clean list.

    Each topic is normalised (:func:`normalize_topic`), then inflectional variants are clustered together (e.g. "knowledge graph"/"knowledge graphs", "coronavirus"/"coronaviruses", "ontology"/"ontologies").
    For each cluster the SHORTEST surface form is kept as canonical (ties broken by frequency, then by first appearance).
    The result is ordered by total cluster frequency (most used first), preserving the frequency-ranked semantics of the topic list.
    """
    if not topics:
        return []

    freq: Counter[str] = Counter()
    order: dict[str, int] = {}
    for raw in topics:
        norm = normalize_topic(raw) if raw else ""
        if not norm:
            continue
        if norm not in order:
            order[norm] = len(order)
        freq[norm] += 1

    uniques = list(order)
    if not uniques:
        return []

    # Union-find over unique topics; only compare within first-character blocks (inflectional/fuzzy variants always share their first character) to keep the pairwise comparison cheap.
    parent: dict[str, str] = {u: u for u in uniques}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    blocks: dict[str, list[str]] = defaultdict(list)
    for u in uniques:
        blocks[u[0]].append(u)
    for group in blocks.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if _is_variant(group[i], group[j]):
                    union(group[i], group[j])

    clusters: dict[str, list[str]] = defaultdict(list)
    for u in uniques:
        clusters[find(u)].append(u)

    ranked: list[tuple[int, int, str]] = []
    for members in clusters.values():
        canonical = min(members, key=lambda m: (len(m), -freq[m], order[m]))
        cluster_freq = sum(freq[m] for m in members)
        cluster_order = min(order[m] for m in members)
        ranked.append((-cluster_freq, cluster_order, canonical))

    ranked.sort()
    return [canonical for _, _, canonical in ranked]


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
def split_labelled_sections(text: str) -> list[tuple[str | None, str]]:
    """Split a portal description into (label, content) sections.

    Each known label terminates the section that precedes it; text remaining after the last label (or the whole text when no label is present) is an unlabelled trailing section with label None.
    Separators left at the start of a section by the previous label are stripped.
    """
    sections: list[tuple[str | None, str]] = []
    pos = 0
    for match in _SECTION_LABEL_REGEX.finditer(text):
        content = _SECTION_SEPARATOR_REGEX.sub("", text[pos : match.start()]).strip()
        sections.append((match.group(0), content))
        pos = match.end()
    tail = _SECTION_SEPARATOR_REGEX.sub("", text[pos:]).strip()
    if tail:
        sections.append((None, tail))
    return sections


def _select_abstract_section(text: str) -> str:
    """Pick the abstract content out of a labelled-section description.

    Preference order (first tier with non-empty content wins):
      1. "(Summary)": the portal's explicit abstract section.
      2. "(General Description)": used instead of Summary by some records.
      3. "(Objectives)" + "(Most relevant results)" joined.
      4. Unlabelled trailing text (records whose abstract carries no label).
    Metadata sections (authors, contact, acknowledgments, merit prose,
    quality indicators...) are always discarded, wherever they appear.
    """
    sections = split_labelled_sections(text)
    if len(sections) == 1 and sections[0][0] is None:
        return text  # no labels at all: the whole text is the candidate

    def collect(*labels: str) -> str:
        return " ".join(
            content for label, content in sections if label in labels and content
        ).strip()

    for candidate in (
        collect(PUBLICATION_SUMMARY_LABEL),
        collect(PUBLICATION_DESCRIPTION_LABEL),
        collect(*PUBLICATION_OBJECTIVES_LABELS),
        collect(None),
    ):
        if candidate:
            return candidate
    return ""


def clean_description(value: str | None) -> str | None:
    """Clean and normalise a publication description.

    Works with the UPM portal's labelled-section format: the description is split into its labelled sections (which appear in ARBITRARY order) and the abstract is selected by its explicit label — "(Summary)", falling back to "(General Description)", then "(Objectives)"/"(Most relevant results)",then unlabelled trailing text.
    Author/contact/acknowledgment/merit sections are discarded, and prefixes, bilingual duplicates, copyright notices and citation metadata are stripped from the selected abstract.
    Returns None when no real content remains.
    """
    if not value:
        return None

    text = value.replace("\xa0", " ").strip()

    if PUBLICATION_MISSING_STATEMENT_REGEX.match(text):
        return None

    text = _select_abstract_section(text)
    # Re-check after selection: the placeholder can also appear INSIDE a
    # labelled section (e.g. "No Statement Available (Summary)").
    if not text or PUBLICATION_MISSING_STATEMENT_REGEX.match(text):
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

    match = PUBLICATION_COPYRIGHT_REGEX.search(text)
    if match:
        prefix = text[: match.start()].strip()
        if prefix:
            text = prefix

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

    Prefers the longest of 'description' / 'DESCRIPCION TFT' and cleans it.
    The cleaner is conservative (it only cuts at explicit section labels) so its result is trusted: when it returns None the record carries no usable abstract (placeholder, affiliation-only, bare URL), and we return an empty string rather than resurfacing that raw junk into the next phases.
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

    return clean_description(raw_description) or ""
