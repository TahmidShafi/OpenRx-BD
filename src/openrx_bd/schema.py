"""Schema definitions and strict validation rules for OpenRx-BD dataset."""

from typing import Any
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# The exact 10 ground-truth columns from Mahadih534/all-Bangladeshi-medicines
# Order, casing, and spacing must match exactly.
GROUND_TRUTH_COLUMNS: tuple[str, ...] = (
    "brand id",
    "brand name",
    "type",
    "slug",
    "dosage form",
    "generic",
    "strength",
    "manufacturer",
    "package container",
    "Package Size",
)


class SchemaViolationError(ValueError):
    """Raised when an upstream dataset violates the locked 10-column schema."""


class RawMedicineRecord(BaseModel):
    """Pydantic model representing an unparsed, raw row from the Hugging Face dataset."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    brand_id: str | int = Field(alias="brand id")
    brand_name: str = Field(alias="brand name")
    type: str = Field(alias="type")
    slug: str = Field(alias="slug")
    dosage_form: str = Field(alias="dosage form")
    generic: str = Field(alias="generic")
    strength: str | None = Field(default=None, alias="strength")
    manufacturer: str = Field(alias="manufacturer")
    package_container: str = Field(alias="package container")
    package_size: str | None = Field(default=None, alias="Package Size")


def validate_columns(columns: list[str]) -> None:
    """Verify that columns match GROUND_TRUTH_COLUMNS exactly in name, casing, and presence."""
    actual_set = set(columns)
    expected_set = set(GROUND_TRUTH_COLUMNS)

    missing = expected_set - actual_set
    unexpected = actual_set - expected_set

    if missing or unexpected:
        error_msgs = []
        if missing:
            error_msgs.append(f"Missing required columns: {sorted(missing)}")
        if unexpected:
            error_msgs.append(f"Found unexpected hallucinated columns: {sorted(unexpected)}")
        raise SchemaViolationError("; ".join(error_msgs))


def validate_dataframe_schema(df: pd.DataFrame) -> None:
    """Strictly assert that a pandas DataFrame adheres to the 10-column schema."""
    validate_columns(list(df.columns))
