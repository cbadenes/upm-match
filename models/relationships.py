"""Script for the edges of the graph database, which represent the relationships between researchers and their publications."""

from __future__ import annotations

from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

EdgeId = Annotated[
    str,
    StringConstraints(min_length=3, max_length=200),
]

NodeRef = Annotated[
    str,
    StringConstraints(min_length=3, max_length=200),
]


class EdgeNode(BaseModel):
    """Base node for graph edges."""

    model_config = ConfigDict(
        populate_by_name=True,
        by_alias=True,
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )
    id: EdgeId = Field(..., description="Unique edge identifier")
    from_node: NodeRef = Field(
        ...,
        alias="_from",
        description="Source node reference",
    )
    to_node: NodeRef = Field(
        ...,
        alias="_to",
        description="Target node reference",
    )


class Authorship(EdgeNode):
    """Represents the authorship relationship between a researcher and a publication.

    The `_from` and `_to` fields use the ArangoDB convention but are agnóstic;
    backends will adapt them as needed (e.g., Memgraph uses :FROM and :TO).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "authorship_pdi_307844_pub_12345",
                "_from": "Pdi/pdi_307844",
                "_to": "Publication/pub_12345",
            }
        }
    )


class Teaching(EdgeNode):
    """Represents the teaching relationship between a researcher and a subject.

    Temporal metadata (start_date, end_date) is stored in the edge itself.
    """

    start_date: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)"
    )
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "teaching_pdi_307844_subj_abc12345",
                "_from": "Pdi/pdi_307844",
                "_to": "Subject/subj_abc12345",
                "start_date": "2022-09-01",
                "end_date": "2024-06-30",
            }
        }
    )


class Works(EdgeNode):
    """Represents the works relationship between a researcher and a position.

    Stores role and temporal information about the researcher's role at a position.
    """

    role: Optional[str] = Field(default=None, description="Role at the position")
    since: Optional[str] = Field(default=None, description="Start date or duration")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "works_pdi_307844_loc_abc12345",
                "_from": "Pdi/pdi_307844",
                "_to": "Position/loc_abc12345",
                "role": "Miembro activo",
                "since": "2015",
            }
        }
    )
