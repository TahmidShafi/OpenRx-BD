"""Pydantic v2 schemas and models for container parsing and dispensing categories."""

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class DispensingCategory(str, Enum):
    """Categorization of dispensing form for pharmaceutical packaging."""

    DIVISIBLE_SOLID = "DIVISIBLE_SOLID"
    INDIVISIBLE_CONTAINER = "INDIVISIBLE_CONTAINER"
    AMBIGUOUS_FORM = "AMBIGUOUS_FORM"


class PackSKU(BaseModel):
    """Parsed single packaging tier / Stock Keeping Unit."""

    model_config = ConfigDict(extra="forbid")

    sku_index: int
    container_type: Optional[str] = None
    measurement_value: Optional[float] = None
    measurement_unit: Optional[str] = None
    unit_price_bdt: Optional[float] = None
    pack_qty: Optional[int] = None
    pack_price_bdt: Optional[float] = None
    effective_per_unit_bdt: float
    invariant_checked: bool = False
    invariant_passed: Optional[bool] = None


class RowParseResult(BaseModel):
    """Result of parsing package container for a single medicine record."""

    model_config = ConfigDict(extra="forbid")

    brand_id: str
    status: Literal[
        "PARSED_VALID",
        "INVARIANT_VIOLATION",
        "ABSTAIN_UNSUPPORTED_TEMPLATE",
        "ABSTAIN_NULL_CONTAINER",
    ]
    dispensing_category: DispensingCategory
    reconciled_total_volume_ml: Optional[float] = None
    skus: list[PackSKU] = Field(default_factory=list)
    abstain_reason: Optional[str] = None
