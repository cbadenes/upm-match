from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Subject(BaseModel):
    """Represents a subject taught by a researcher.

    The `id` field is agnóstic and will be mapped appropriately in each backend.
    The `title` field may be mapped to `name` in certain backends.
    The `type` field is the category or level.
    Temporal information (start_date, end_date) is stored in the teaching edge, not here.
    """

    id: str = Field(
        ..., description="Unique subject identifier (e.g., 'subj_abc12345')"
    )
    title: str = Field(..., alias="name", description="Subject name or title")
    type: str = Field(
        ..., description="Subject type or category (e.g., 'Docencia Oficial Nacional')"
    )
    start_date: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "id": "subj_abc12345",
                "name": "Algoritmos Avanzados",
                "type": "Docencia Oficial Nacional",
                "start_date": "2022-09-01",
                "end_date": "2024-06-30",
            }
        },
    )
