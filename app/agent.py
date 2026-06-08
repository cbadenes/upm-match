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
        model=Ollama(id=_OLLAMA_MODEL, host=_OLLAMA_HOST),
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
        model=Ollama(id=_OLLAMA_MODEL, host=_OLLAMA_HOST),
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
