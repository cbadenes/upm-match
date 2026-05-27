from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

LastYearProjectId = Annotated[str, StringConstraints(pattern=r"^tft_\d+$")]


class LastYearProject(BaseModel):
    """Represents a thesis or final-degree supervision record."""

    id: LastYearProjectId = Field(
        ..., description="Unique last year project identifier (e.g., 'tft_67901')"
    )
    oa_id: Optional[str] = Field(
        default=None, description="Open Access identifier (e.g., OA UPM URL)"
    )
    title: str = Field(..., alias="name", description="Last year project title")
    description: str = Field(
        default="", description="Last year project description from the AJAX dataset"
    )
    authors: list[str] = Field(default_factory=list, description="List of author names")
    keywords: list[str] = Field(
        default_factory=list, description="Last year project keywords/subjects"
    )
    year: int = Field(..., description="Last year project year")
    type: str = Field(..., description="Last year project subtype")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "id": "tft_67901",
                "oa_id": "https://oa.upm.es/67901/",
                "title": "Explainable QA over KG",
                "description": "Trabajo fin de máster sobre ...",
                "authors": ["Inés Guillén", "Rafael"],
                "keywords": ["informática", "knowledge graphs"],
                "year": 2021,
                "type": "DEA, máster, etc.",
            }
        },
    )
