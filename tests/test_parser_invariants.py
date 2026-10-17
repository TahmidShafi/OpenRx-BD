"""Unit tests for package container parser invariants, edge cases, and dimensional reconciliation."""

import pytest
from src.parser.container_parser import (
    classify_dispensing_category,
    parse_concentration_mg_per_ml,
    parse_container,
)
from src.parser.schemas import DispensingCategory, RowParseResult


def test_1_a_cal_d_double_comma_multi_sku():
    """Case 1: A-Cal D multi-SKU separated by double comma."""
    raw = "Unit Price: ৳ 6.50,(15's pack: ৳ 97.50),,Unit Price: ৳ 6.50,(30's pack: ৳ 195.00),"
    res = parse_container(
        brand_id="14399",
        package_container=raw,
        dosage_form="Tablet",
        strength="500 mg+200 IU",
    )
    assert res.status == "PARSED_VALID"
    assert res.dispensing_category == DispensingCategory.DIVISIBLE_SOLID
    assert len(res.skus) == 2

    # SKU 0: 15's pack
    sku0 = res.skus[0]
    assert sku0.sku_index == 0
    assert sku0.unit_price_bdt == 6.50
    assert sku0.pack_qty == 15
    assert sku0.pack_price_bdt == 97.50
    assert sku0.invariant_checked is True
    assert sku0.invariant_passed is True

    # SKU 1: 30's pack
    sku1 = res.skus[1]
    assert sku1.sku_index == 1
    assert sku1.unit_price_bdt == 6.50
    assert sku1.pack_qty == 30
    assert sku1.pack_price_bdt == 195.00
    assert sku1.invariant_checked is True
    assert sku1.invariant_passed is True


def test_2_5_fluril_cross_column_dimensional_reconciliation():
    """Case 2: 5-Fluril cross-column reconciliation (25 mg/ml vs 250 mg vial)."""
    raw = "250 mg vial: ৳ 55.00,500 mg vial: ৳ 95.00"
    res = parse_container(
        brand_id="105",
        package_container=raw,
        dosage_form="IV Injection or Infusion",
        strength="25 mg/ml",
    )
    assert res.status == "PARSED_VALID"
    assert res.dispensing_category == DispensingCategory.INDIVISIBLE_CONTAINER
    assert len(res.skus) == 2

    # Dimensional reconciliation: 250 mg / 25 mg/ml = 10.0 ml
    assert res.reconciled_total_volume_ml == 10.0

    assert res.skus[0].measurement_value == 250.0
    assert res.skus[0].measurement_unit == "mg"
    assert res.skus[0].container_type == "vial"
    assert res.skus[0].effective_per_unit_bdt == 55.00

    assert res.skus[1].measurement_value == 500.0
    assert res.skus[1].measurement_unit == "mg"
    assert res.skus[1].container_type == "vial"
    assert res.skus[1].effective_per_unit_bdt == 95.00


def test_3_apelin_multi_bottle_herbal_null_strength():
    """Case 3: Apelin herbal multi-bottle with null strength."""
    raw = "100 ml bottle: ৳ 60.00,450 ml bottle: ৳ 200.00"
    res = parse_container(
        brand_id="30450",
        package_container=raw,
        dosage_form="Syrup",
        strength=None,
    )
    assert res.status == "PARSED_VALID"
    assert res.dispensing_category == DispensingCategory.INDIVISIBLE_CONTAINER
    assert len(res.skus) == 2

    # Direct volume recognized from primary bottle
    assert res.reconciled_total_volume_ml == 100.0
    assert res.skus[0].container_type == "bottle"
    assert res.skus[0].measurement_value == 100.0
    assert res.skus[0].measurement_unit == "ml"
    assert res.skus[0].effective_per_unit_bdt == 60.00

    assert res.skus[1].container_type == "bottle"
    assert res.skus[1].measurement_value == 450.0
    assert res.skus[1].measurement_unit == "ml"
    assert res.skus[1].effective_per_unit_bdt == 200.00


def test_4_corrupted_math_invariant_violation():
    """Case 4: Corrupted math must flag INVARIANT_VIOLATION."""
    # 6.50 * 10 = 65 != 99
    raw = "Unit Price: ৳ 6.50,(10's pack: ৳ 99.00)"
    res = parse_container(
        brand_id="99901",
        package_container=raw,
        dosage_form="Tablet",
        strength="500 mg",
    )
    assert res.status == "INVARIANT_VIOLATION"
    assert len(res.skus) == 1
    assert res.skus[0].invariant_checked is True
    assert res.skus[0].invariant_passed is False
    assert "Multiplicative invariant failed" in (res.abstain_reason or "")


def test_5_unseen_gibberish_abstains():
    """Case 5: Unseen gibberish must flag ABSTAIN_UNSUPPORTED_TEMPLATE."""
    raw = "XYZ Random 999 Foobar No Taka"
    res = parse_container(
        brand_id="99902",
        package_container=raw,
        dosage_form="Tablet",
    )
    assert res.status == "ABSTAIN_UNSUPPORTED_TEMPLATE"
    assert len(res.skus) == 0


def test_6_null_package_container_abstains():
    """Case 6: Null container must flag ABSTAIN_NULL_CONTAINER."""
    res = parse_container(
        brand_id="99903",
        package_container=None,
        dosage_form="Syrup",
    )
    assert res.status == "ABSTAIN_NULL_CONTAINER"
    assert res.abstain_reason == "Package container is null or empty"


def test_7_empty_whitespace_container_abstains():
    """Case 7: Whitespace container must flag ABSTAIN_NULL_CONTAINER."""
    res = parse_container(
        brand_id="99904",
        package_container="   \n\t  ",
        dosage_form="Syrup",
    )
    assert res.status == "ABSTAIN_NULL_CONTAINER"


def test_8_price_unavailable_abstains():
    """Case 8: 'Price Unavailable' explicitly abstains."""
    res = parse_container(
        brand_id="99905",
        package_container="Price Unavailable",
        dosage_form="Tablet",
    )
    assert res.status == "ABSTAIN_UNSUPPORTED_TEMPLATE"


def test_9_not_for_sale_abstains():
    """Case 9: 'Not for sale' explicitly abstains."""
    res = parse_container(
        brand_id="99906",
        package_container="Not for sale",
        dosage_form="Injection",
    )
    assert res.status == "ABSTAIN_UNSUPPORTED_TEMPLATE"


def test_10_vial_with_pack_pricing():
    """Case 10: Vial packaging with pack info."""
    raw = "500 mg vial: ৳ 28.43,(5's pack: ৳ 142.15),"
    res = parse_container(
        brand_id="6174",
        package_container=raw,
        dosage_form="Injection",
        strength="500 mg/vial",
    )
    assert res.status == "PARSED_VALID"
    assert len(res.skus) == 1
    sku = res.skus[0]
    assert sku.container_type == "vial"
    assert sku.measurement_value == 500.0
    assert sku.measurement_unit == "mg"
    assert sku.unit_price_bdt == 28.43
    assert sku.pack_qty == 5
    assert sku.pack_price_bdt == 142.15
    assert sku.invariant_passed is True


def test_11_single_tube_pricing():
    """Case 11: Single tube ointment."""
    raw = "15 gm tube: ৳ 45.00"
    res = parse_container(
        brand_id="99907",
        package_container=raw,
        dosage_form="Ointment",
    )
    assert res.status == "PARSED_VALID"
    assert len(res.skus) == 1
    assert res.skus[0].container_type == "tube"
    assert res.skus[0].measurement_value == 15.0
    assert res.skus[0].measurement_unit == "gm"
    assert res.skus[0].effective_per_unit_bdt == 45.00


def test_12_metered_dose_inhaler():
    """Case 12: Inhaler metered doses."""
    raw = "200 metered doses: ৳ 350.00"
    res = parse_container(
        brand_id="99908",
        package_container=raw,
        dosage_form="Inhaler",
    )
    assert res.status == "PARSED_VALID"
    assert len(res.skus) == 1
    assert res.skus[0].measurement_value == 200.0
    assert res.skus[0].effective_per_unit_bdt == 350.00


def test_13_multi_sku_with_one_corrupted_fails_row():
    """Case 13: Row with 1 valid and 1 corrupted SKU fails row invariant."""
    raw = "Unit Price: ৳ 5.00,(10's pack: ৳ 50.00),,Unit Price: ৳ 2.00,(10's pack: ৳ 80.00)"
    res = parse_container(
        brand_id="99909",
        package_container=raw,
        dosage_form="Tablet",
    )
    assert res.status == "INVARIANT_VIOLATION"
    assert len(res.skus) == 2
    assert res.skus[0].invariant_passed is True
    assert res.skus[1].invariant_passed is False


def test_14_compound_syrup_concentration_reconciliation():
    """Case 14: Compound syrup concentration reconciliation."""
    # (10 mg+30 mg)/5 ml -> 40 / 5 = 8.0 mg/ml
    # 80 mg bottle -> 80 / 8.0 = 10.0 ml
    res = parse_container(
        brand_id="99910",
        package_container="80 mg bottle: ৳ 90.00",
        dosage_form="Syrup",
        strength="(10 mg+30 mg)/5 ml",
    )
    assert res.status == "PARSED_VALID"
    assert res.reconciled_total_volume_ml == 10.0


def test_15_suspension_concentration_reconciliation():
    """Case 15: Suspension concentration reconciliation."""
    # 125 mg/5 ml -> 25.0 mg/ml
    # 250 mg bottle -> 250 / 25.0 = 10.0 ml
    res = parse_container(
        brand_id="99911",
        package_container="250 mg bottle: ৳ 45.00",
        dosage_form="Suspension",
        strength="125 mg/5 ml",
    )
    assert res.status == "PARSED_VALID"
    assert res.reconciled_total_volume_ml == 10.0


def test_16_dispensing_categories_broad_coverage():
    """Case 16: Verify dispensing category logic for diverse dosage forms."""
    assert classify_dispensing_category("Tablet") == DispensingCategory.DIVISIBLE_SOLID
    assert classify_dispensing_category("Capsule (Delayed Release)") == DispensingCategory.DIVISIBLE_SOLID
    assert classify_dispensing_category("IV Injection") == DispensingCategory.INDIVISIBLE_CONTAINER
    assert classify_dispensing_category("Eye Drop") == DispensingCategory.INDIVISIBLE_CONTAINER
    assert classify_dispensing_category("Skin Cream") == DispensingCategory.INDIVISIBLE_CONTAINER
    assert classify_dispensing_category("Kit") == DispensingCategory.AMBIGUOUS_FORM
