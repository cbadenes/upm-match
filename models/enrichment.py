from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ResearcherPosition(BaseModel):
    """Position held by a researcher."""

    id: str
    name: Optional[str] = None
    entity: Optional[str] = None
    role: Optional[str] = None
    since: Optional[int] = None

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearcherPublication(BaseModel):
    """Publication linked to a researcher."""

    id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    type: Optional[str] = None
    year: Optional[int] = None
    source: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    coauthorship: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearcherSubject(BaseModel):
    """Subject taught by a researcher."""

    name: Optional[str] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearcherLastYearProject(BaseModel):
    """Last year project supervised by a researcher."""

    id: str
    title: Optional[str] = None
    abstract: Optional[str] = None
    type: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    year: Optional[int] = None

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResearcherEnrichmentDocument(BaseModel):
    """Pydantic contract for the researcher enrichment document."""

    id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    first_year: Optional[int] = None
    topics: list[str] = Field(default_factory=list)
    positions: list[ResearcherPosition] = Field(default_factory=list)
    publications: list[ResearcherPublication] = Field(default_factory=list)
    subjects: list[ResearcherSubject] = Field(default_factory=list)
    last_year_projects: list[ResearcherLastYearProject] = Field(default_factory=list)
    generated_at: datetime
    # Populated by 06_create_summaries: rolling biography snapshot per year.
    history: dict[str, Any] = Field(default_factory=dict)
    # Populated by 07_index_summaries: the recent-weighted fused "current" profile
    # and its expertise domains, written back to the top level of the document.
    summary: Optional[str] = None
    domains: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
