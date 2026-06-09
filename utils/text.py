"""Text normalisation and cleaning utilities shared across the pipeline."""

import re
from collections import Counter, defaultdict
from unicodedata import normalize as unicodedata_normalize

# Markers used by clean_description
#
# The UPM portal serialises a publication record as a sequence of labelled sections, each section TERMINATED by a parenthesised label, e.g.:
#
#   <authors> (Authors with Filk); <abstract> (Summary); <contact> (Author of contact); <recognitions> (Recognitions linked to item)
#
# So the abstract is the text that ENDS with "(Summary)" (or, when there is no explicit summary section, the text before the first of the other labels).
# "(Authors with Filk)" is special: it terminates a LEADING author block, so everything up to and including it must be dropped.

PUBLICATION_AUTHORS_BLOCK_MARKER = "(Authors with Filk)"

# Section labels that mark the END of the abstract: cut from the earliest one.
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
    "(Author of contact)",
    "(Recognitions linked to item)",
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

    Steps: lowercase, strip a trailing possessive ("'s"), NFKD → ASCII, trim,
    remove trailing period, collapse spaces. Plural/inflection merging is NOT
    done here (a per-string view cannot tell "analysis" from a real plural) — use
    :func:`canonicalize_topics` on the whole list for that.
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

    Handles -s, -es and -y/-ies. This is enough to merge "graph"/"graphs",
    "disease"/"diseases", "box"/"boxes", "ontology"/"ontologies" and even
    "coronavirus"/"coronaviruses" (singular already ends in 's'). A short fuzzy
    fallback catches light typos / diacritic leftovers.
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

    Compared token by token: they must have the same number of tokens, every
    token must match or be a plural/inflection of its counterpart, and at least
    one token must differ (otherwise they are identical). The equal-token-count
    rule keeps distinct concepts apart ("language model" vs "large language
    model", "topic model" vs "dynamic topic model").
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
def clean_description(value: str | None) -> str | None:
    """Clean and normalise a publication description.

    Works with the UPM portal's labelled-section format: it drops a leading author block (terminated by "(Authors with Filk)"), keeps the abstract up to the first section label and strips prefixes, bilingual duplicates and copyright notices.
    Returns None when no real content remains.

    The cleaning is intentionally conservative: it only cuts at explicit section labels / copyright notices, never on loose heuristics, so a real abstract is never silently truncated.
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

    def earliest_marker(haystack: str) -> int | None:
        """Index of the first abstract-ending section label, or None."""
        found = None
        for marker in PUBLICATION_TRAILING_MARKERS:
            marker_index = haystack.find(marker)
            if marker_index != -1 and (found is None or marker_index < found):
                found = marker_index
        return found

    # An author block ("(Authors with Filk)") only leads when it appears BEFORE the abstract-ending label.
    # In many records the order is reversed (<abstract> (Summary); <authors> (Authors with Filk)), so stripping unconditionally would wrongly drop the abstract.
    authors_index = text.find(PUBLICATION_AUTHORS_BLOCK_MARKER)
    end_index = earliest_marker(text)
    if authors_index != -1 and (end_index is None or authors_index < end_index):
        cut = authors_index + len(PUBLICATION_AUTHORS_BLOCK_MARKER)
        text = re.sub(r"^\s*;\s*", "", text[cut:]).strip()
        end_index = earliest_marker(text)

    match = PUBLICATION_COPYRIGHT_REGEX.search(text)
    if match:
        prefix = text[: match.start()].strip()
        if prefix:
            text = prefix
        end_index = earliest_marker(text)

    if end_index is not None:
        text = text[:end_index]

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
