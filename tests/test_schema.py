"""Unit tests for strict 10-column schema locking and anti-hallucination constraints."""

import pandas as pd
import pytest
from openrx_bd.schema import (
    GROUND_TRUTH_COLUMNS,
    RawMedicineRecord,
    SchemaViolationError,
    validate_columns,
    validate_dataframe_schema,
)


def test_ground_truth_columns_exact_spec():
    """Ensure the 10 columns match the exact Hugging Face dataset definition."""
    expected = (
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
    assert GROUND_TRUTH_COLUMNS == expected
    assert len(GROUND_TRUTH_COLUMNS) == 10


def test_validate_columns_success():
    """Correct column list passes validation."""
    validate_columns(list(GROUND_TRUTH_COLUMNS))


def test_validate_columns_missing_column():
    """Missing column triggers SchemaViolationError."""
    bad_columns = [c for c in GROUND_TRUTH_COLUMNS if c != "package container"]
    with pytest.raises(SchemaViolationError, match="Missing required columns.*package container"):
        validate_columns(bad_columns)


def test_validate_columns_hallucinated_column():
    """Hallucinated column triggers SchemaViolationError."""
    bad_columns = list(GROUND_TRUTH_COLUMNS) + ["price", "discounted_price"]
    with pytest.raises(SchemaViolationError, match="Found unexpected hallucinated columns"):
        validate_columns(bad_columns)


def test_raw_medicine_record_valid():
    """Valid record parses cleanly into RawMedicineRecord."""
    sample = {
        "brand id": "4077",
        "brand name": "A-Cold",
        "type": "allopathic",
        "slug": "a-coldsyrup4-mg5-ml",
        "dosage form": "Syrup",
        "generic": "Bromhexine Hydrochloride",
        "strength": "4 mg/5 ml",
        "manufacturer": "ACME Laboratories Ltd.",
        "package container": "100 ml bottle: ৳ 40.12,",
        "Package Size": "",
    }
    rec = RawMedicineRecord.model_validate(sample)
    assert rec.brand_name == "A-Cold"
    assert rec.dosage_form == "Syrup"
    assert rec.package_container == "100 ml bottle: ৳ 40.12,"


def test_raw_medicine_record_rejects_hallucinated_field():
    """Pydantic model rejects unknown/hallucinated fields (extra='forbid')."""
    sample = {
        "brand id": "4077",
        "brand name": "A-Cold",
        "type": "allopathic",
        "slug": "a-coldsyrup4-mg5-ml",
        "dosage form": "Syrup",
        "generic": "Bromhexine Hydrochloride",
        "strength": "4 mg/5 ml",
        "manufacturer": "ACME Laboratories Ltd.",
        "package container": "100 ml bottle: ৳ 40.12,",
        "Package Size": "",
        "hallucinated_api_cost": 10.5,
    }
    with pytest.raises(Exception):
        RawMedicineRecord.model_validate(sample)
