from pydantic import BaseModel, Field


class WeightedPoseVariant(BaseModel):
    text: str
    weight: float = Field(default=1.0, gt=0)


class PoseLayout(BaseModel):
    id: str
    name: str = ""
    weight: float = Field(default=1.0, gt=0)
    required: list[str] = Field(default_factory=list)
    variants: dict[str, list[WeightedPoseVariant]] = Field(default_factory=dict)
    optional_group_chance: float | None = Field(default=None, ge=0.0, le=1.0)


class PosePreset(BaseModel):
    id: str
    name: str
    description: str = ""
    search_terms: list[str] = Field(default_factory=list)
    compatible_contexts: list[str] = Field(default_factory=list)
    incompatible_contexts: list[str] = Field(default_factory=list)
    required_context_any: list[str] = Field(default_factory=list)
    orientation: list[str] = Field(default_factory=lambda: ["any"])
    layouts: list[PoseLayout]
    enabled: bool = True
    selection_weight: float = Field(default=1.0, gt=0)
    optional_group_chance: float = Field(default=0.7, ge=0.0, le=1.0)
