from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

PublicationId = Annotated[str, StringConstraints(pattern=r"^pub_\d+$")]


class Publication(BaseModel):
    """Represents a publication authored by a researcher.

    The `id` field is agnóstic and will be mapped appropriately in each backend.
    The `title` field may be mapped to `name` in certain backends.
    URLs can be a single string or a list of strings.
    """

    id: PublicationId = Field(
        ..., description="Unique publication identifier (e.g., 'pub_12345678')"
    )
    oa_id: Optional[str] = Field(
        default=None, description="Open Access identifier (e.g., OA UPM URL)"
    )
    title: str = Field(..., alias="name", description="Publication title")
    url: str | list[Optional[str]] = Field(
        default=None, description="Publication URL(s)"
    )
    year: int = Field(..., description="Publication year")
    type: str = Field(
        ..., description="Publication type (e.g., 'Artículo', 'Conferencia')"
    )
    authors: list[str] = Field(default_factory=list, description="List of author names")
    keywords: list[str] = Field(
        default_factory=list, description="Publication keywords/subjects"
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
                "id": "pub_12345678",
                "oa_id": "https://oa.upm.es/12345678/",
                "title": "Machine Learning Applications in Healthcare",
                "url": "https://doi.org/10.1234/example",
                "year": 2023,
                "type": "Artículo",
                "authors": ["John Doe", "Jane Smith"],
                "keywords": ["machine learning", "healthcare", "AI"],
            }
        },
    )
