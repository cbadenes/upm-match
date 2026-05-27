from __future__ import annotations

from .last_year_project import LastYearProject
from .pdi import Researcher
from .position import Position
from .publication import Publication
from .relationships import Authorship, Teaching, Tutoring, Works
from .subject import Subject

__all__ = [
    # Node models
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
