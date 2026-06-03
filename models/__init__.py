from __future__ import annotations

from .enrichment import (
    ResearcherEnrichmentDocument,
    ResearcherLastYearProject,
    ResearcherPosition,
    ResearcherPublication,
    ResearcherSubject,
)
from .last_year_project import LastYearProject
from .pdi import Researcher
from .position import Position
from .publication import Publication
from .relationships import Authorship, Teaching, Tutoring, Works
from .subject import Subject

__all__ = [
    # Node models
    "ResearcherEnrichmentDocument",
    "ResearcherLastYearProject",
    "ResearcherPosition",
    "ResearcherPublication",
    "ResearcherSubject",
    "Researcher",
    "Publication",
    "Position",
    "Subject",
    "LastYearProject",
    # Edge models
    "Authorship",
    "Teaching",
    "Tutoring",
    "Works",
]
