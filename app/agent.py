"""Agno-based agents powered by a local Ollama model.

Two agents are provided:
  - extract_keywords: extracts compact English keyword lists from a research idea.
  - enrich_query: translates and expands a research idea into an English paragraph optimised for dense vector search against researcher profile summaries.
"""

from __future__ import annotations

import os

from agno.agent import Agent
from agno.models.ollama import Ollama
from pydantic import BaseModel, Field

# Build the host from OLLAMA_HOST if set, otherwise derive it from OLLAMA_PORT
_OLLAMA_PORT = os.environ.get("OLLAMA_PORT", "11434")
_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", f"http://localhost:{_OLLAMA_PORT}")
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
_OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.1"))
_OLLAMA_OPTIONS = {"temperature": _OLLAMA_TEMPERATURE}

# Crafted to steer models away from verbose keyword phrases, which is the most common failure mode when models are asked to generate keyword lists.
_DESCRIPTION = (
    "You are a research keyword extractor for an academic advisor matching system at the university. "
    "Your output is used directly as search queries against an indexed academic database, so precision and conciseness are critical."
)

_INSTRUCTIONS = [
    "Extract exactly two keyword lists from the given research idea.",
    "RULES - violating any rule degrades search quality:",
    "  1. English ONLY and lowercase.",
    "  2. Each keyword: 1–4 words. No longer.",
    "  3. No abbreviations, no parentheses, no slashes, no punctuation.",
    "  4. Format like the 'Keywords' field in an IEEE / ACM / Elsevier paper.",
    "  5. No duplicates between the two lists.",
    "CORRECT examples: 'machine learning', 'question generation', 'transfer learning', 'bloom's taxonomy'.",
    "INCORRECT examples: 'AI-Powered Assessment Tools', 'LLM-based QG (Question Generation)', 'Adaptive Systems using ML'.",
    "List GENERAL - 5 to 8 items: umbrella research disciplines and domains.",
    "List SPECIFIC - 8 to 12 items: precise methods, algorithms, architectures, or applications.",
    "Even though the input idea may be verbose and in any language, your output MUST be concise and in English to work well as search keywords.",
    "Focus on extracting the core research concepts and techniques, not on rephrasing the input idea.",
]


class KeywordSet(BaseModel):
    """Structured output produced by the keyword extraction agent."""

    general: list[str] = Field(
        description="5–8 broad research domain keywords, each 1–4 words, lowercase English.",
    )
    specific: list[str] = Field(
        description="8–12 specific technique or method keywords, each 1–4 words, lowercase English.",
    )


def extract_keywords(idea: str) -> KeywordSet:
    """Run the keyword extraction agent on a research idea.

    :param idea: Free-text description of the research idea (any language).
    :return: A KeywordSet with two lists of compact English keywords.
    :raises RuntimeError: If the agent fails to produce a valid structured response.
    """
    agent = Agent(
        model=Ollama(id=_OLLAMA_MODEL, host=_OLLAMA_HOST, options=_OLLAMA_OPTIONS),
        description=_DESCRIPTION,
        instructions=_INSTRUCTIONS,
        output_schema=KeywordSet,
        markdown=False,
    )
    response = agent.run(idea)

    # Agno swallows connection / model errors and returns them as plain strings instead of raising exceptions - detect and re-raise explicitly.
    if isinstance(response.content, str):
        raise RuntimeError(f"Ollama error: {response.content}")

    if isinstance(response.content, KeywordSet):
        return response.content

    # The field output_schema may return a dict when the model uses JSON mode without tool calling
    if isinstance(response.content, dict):
        return KeywordSet.model_validate(response.content)

    raise RuntimeError(f"Agent returned unexpected type: {type(response.content)}")


# Query enrichment agent
_ENRICH_DESCRIPTION = (
    "You are a research query optimizer for an academic advisor matching system. "
    "Your output is used to search a database of researcher profile summaries written in English."
)

_ENRICH_INSTRUCTIONS = [
    "You receive a research description written by a student (in any language).",
    "Return a single `text` field: a fluent English paragraph of 3–5 sentences.",
    "RULES — breaking any rule degrades search quality:",
    "  1. Translate faithfully: keep every concept the student mentioned.",
    "  2. You may add closely related technical terminology that a researcher in this area would recognise.",
    "  3. NEVER add topics, applications, or domains not clearly implied by the student's text.",
    "  4. Write in the third person: 'This research focuses on ...'",
    "  5. Continuous prose only — no bullet points, no lists, no headings.",
    "  6. Do not reference the student, the university, or the thesis format.",
]


class EnrichedQuery(BaseModel):
    """Structured output produced by the query enrichment agent."""

    text: str = Field(
        description=(
            "3–5 sentence English paragraph describing the research topic. "
            "Faithful translation with closely related terminology added. No invented topics."
        )
    )


def enrich_query(idea: str) -> str:
    """Translate and enrich a student's research idea for dense vector search.

    :param idea: Free-text research description in any language.
    :return: An enriched English paragraph suitable for embedding against researcher summaries.
    :raises RuntimeError: If the agent fails to produce a valid structured response.
    """
    agent = Agent(
        model=Ollama(id=_OLLAMA_MODEL, host=_OLLAMA_HOST, options=_OLLAMA_OPTIONS),
        description=_ENRICH_DESCRIPTION,
        instructions=_ENRICH_INSTRUCTIONS,
        output_schema=EnrichedQuery,
        markdown=False,
    )
    response = agent.run(idea)

    if isinstance(response.content, str):
        raise RuntimeError(f"Ollama error: {response.content}")

    if isinstance(response.content, EnrichedQuery):
        return response.content.text

    if isinstance(response.content, dict):
        return EnrichedQuery.model_validate(response.content).text

    raise RuntimeError(f"Agent returned unexpected type: {type(response.content)}")


# Student-case generation agent (used by the PDI questionnaire, section 5)
_CASE_DESCRIPTION = (
    "You write realistic student research queries to evaluate an academic supervisor "
    "recommender. You are given several researcher profiles that are ALL suitable "
    "supervisors for the same hypothetical thesis project."
)
_CASE_INSTRUCTIONS = [
    "Write ONE plausible research idea that a university student (TFG/TFM) might submit "
    "when looking for a thesis supervisor, such that the given researchers would all be good matches.",
    "RULES:",
    "  1. Write the `query` in SPANISH, 2 to 4 sentences, in natural language as a student would phrase it.",
    "  2. Make it realistic and concrete, but broad enough to fit the shared expertise of the group.",
    "  3. Do NOT mention researcher names, institutions, or that this text was generated.",
    "  4. Also return `keywords`: 5 to 10 lowercase ENGLISH keywords (1-3 words each) capturing the core topics.",
    "  5. Base the idea ONLY on the expertise shown; do not invent unrelated fields.",
]


class StudentCase(BaseModel):
    """A generated student query plus keywords, for the recommendation-evaluation section."""

    query: str = Field(
        description="2–4 sentence plausible student research idea, in Spanish."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="5–10 lowercase English keywords capturing the topic, for matching.",
    )


def generate_student_case(profiles: list[dict]) -> StudentCase:
    """Generate a plausible student query for a cluster of similar researcher profiles.

    :param profiles: list of {"summary": str, "domains": list[str]} for the group.
    :return: StudentCase with a Spanish query and English keywords.
    :raises RuntimeError: If the agent fails to produce a valid structured response.
    """
    blocks = []
    for i, p in enumerate(profiles, 1):
        domains = ", ".join(p.get("domains") or [])
        summary = (p.get("summary") or "").strip()
        blocks.append(f"Researcher {i}:\nDomains: {domains}\nProfile: {summary}")
    prompt = (
        "These researchers are all suitable supervisors for the SAME student project:\n\n"
        + "\n\n".join(blocks)
        + "\n\nWrite the student query and keywords."
    )

    agent = Agent(
        model=Ollama(id=_OLLAMA_MODEL, host=_OLLAMA_HOST, options=_OLLAMA_OPTIONS),
        description=_CASE_DESCRIPTION,
        instructions=_CASE_INSTRUCTIONS,
        output_schema=StudentCase,
        markdown=False,
    )
    response = agent.run(prompt)

    if isinstance(response.content, str):
        raise RuntimeError(f"Ollama error: {response.content}")
    if isinstance(response.content, StudentCase):
        return response.content
    if isinstance(response.content, dict):
        return StudentCase.model_validate(response.content)

    raise RuntimeError(f"Agent returned unexpected type: {type(response.content)}")
