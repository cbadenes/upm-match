from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

ResearcherId = Annotated[str, StringConstraints(pattern=r"^pdi_\d{6}$")]
EmailStr = Annotated[str, StringConstraints(pattern=r"^[\w\.-]+@[\w\.-]+$")]
TopicStr = Annotated[str, StringConstraints(min_length=1)]


class Researcher(BaseModel):
    """Represents a researcher profile with personal data and research metadata.

    The `id` field is agnostic and will be mapped to `_key` in ArangoDB or used directly in Memgraph.
    External IDs are stored as a dictionary where keys are ID types.
    """

    id: ResearcherId = Field(
        ...,
        alias="_key",
        description="Unique researcher identifier (e.g., 'pdi_307844')",
    )
    full_name: str = Field(..., alias="name", description="Full name of the researcher")
    first_name: Optional[str] = Field(default=None, description="First name")
    last_name: Optional[str] = Field(default=None, description="Last name")
    email: Optional[EmailStr] = Field(default=None, description="Email address")
    external_ids: dict[str, Optional[str]] = Field(
        default_factory=dict,
        description="External IDs (orcid_id, scopus_id, researcher_id, etc.)",
    )
    topics: list[TopicStr] = Field(
        default_factory=list,
        description="Research topics (obtained directly from UPM portal, not inferred from publications)",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "id": "pdi_307844",
                "full_name": "Juan García López",
                "first_name": "Juan",
                "last_name": "García López",
                "email": "j.garcia@upm.es",
                "external_ids": {
                    "orcid_id": "0000-0001-2345-6789",
                    "scopus_id": "1234567890",
                },
                "topics": ["machine learning", "computer vision", "deep learning"],
            }
        },
    )
