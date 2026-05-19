from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Position(BaseModel):
    """Represents a position held by a researcher at an institution.

    The `id` field is agnóstic and will be mapped appropriately in each backend.
    Temporal information (role, since) is stored in the edge relationship, not in the node.
    """

    id: str = Field(
        ..., description="Unique position identifier (e.g., 'loc_abc12345')"
    )
    name: Optional[str] = Field(
        default=None, description="Position title or entity name"
    )
    entity: Optional[str] = Field(
        default=None, description="Institution/organization name"
    )
    role: Optional[str] = Field(default=None, description="Role within the position")
    since: Optional[str] = Field(
        default=None, description="Start date or duration string"
    )
    source_id: Optional[str] = Field(
        default=None, description="Source system ID (e.g., UPM portal ID)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "loc_abc12345",
                "name": "Profesor Titular de Universidad",
                "entity": "Departamento de Sistemas Informáticos",
                "role": "Miembro activo",
                "since": "2015",
                "source_id": "12345",
            }
        }
    )
