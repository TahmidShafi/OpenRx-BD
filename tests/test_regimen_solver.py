"""Unit and regression tests for pharmaceutical regimen solver, knapsack dispensing, and polypharmacy safety."""

import polars as pl
import pytest
from src.optimizer.regimen_solver import (
    PrescriptionItem,
    RegimenOptimizer,
    solve_divisible_solid_dispensing,
    solve_indivisible_knapsack,
)
from src.parser.schemas import PackSKU


def test_unbounded_knapsack_safi_bottles():
    """Verify exact unbounded knapsack combines 450 ml + 225 ml bottles to cover 600 ml at minimum cost."""
    containers = [
        (100.0, 65.0, "100 ml bottle"),
        (225.0, 120.0, "225 ml bottle"),
        (450.0, 175.0, "450 ml bottle"),
    ]
    # Target: 600 ml
    # Option 1: 6x 100ml = 600ml @ 390 BDT
    # Option 2: 3x 225ml = 675ml @ 360 BDT
    # Option 3: 2x 450ml = 900ml @ 350 BDT
    # Optimum: 1x 450ml (175) + 1x 225ml (120) = 675ml @ 295 BDT (waste = 75 ml)
    plan = solve_indivisible_knapsack(600.0, containers, unit_name="ml")
    assert plan.total_cost_bdt == 295.0
    assert plan.total_units_supplied == 675.0
    assert plan.leftover_waste == 75.0
    assert plan.strategy == "INDIVISIBLE_CONTAINER"
    assert len(plan.container_breakdown) == 2


def test_unbounded_knapsack_tie_break_minimizes_waste():
    """Verify knapsack tie-breaking minimizes waste when costs are identical."""
    containers = [
        (50.0, 100.0, "50 ml bottle"),
        (60.0, 100.0, "60 ml bottle"),
    ]
    # Target: 50 ml
    # Both 50 ml and 60 ml cost 100.0 BDT. The solver must pick 50 ml (0 waste).
    plan = solve_indivisible_knapsack(50.0, containers, unit_name="ml")
    assert plan.total_cost_bdt == 100.0
    assert plan.total_units_supplied == 50.0
    assert plan.leftover_waste == 0.0


def test_unbounded_knapsack_small_dose_drops():
    """Verify fractional and small dosing for eye drops."""
    containers = [
        (5.0, 30.0, "5 ml dropper bottle"),
    ]
    # Need 12.5 ml -> ceil(12.5 / 5) = 3 bottles = 15.0 ml @ 90 BDT (waste: 2.5 ml)
    plan = solve_indivisible_knapsack(12.5, containers, unit_name="ml")
    assert plan.total_cost_bdt == 90.0
    assert plan.total_units_supplied == 15.0
    assert plan.leftover_waste == 2.5


def test_divisible_solid_cut_strip_retail_dispensing():
    """Verify cut blister strip dispensing in Bangladesh retail pharmacies."""
    skus = [
        PackSKU(
            sku_index=0,
            unit_price_bdt=5.0,
            pack_qty=100,
            pack_price_bdt=500.0,
            effective_per_unit_bdt=5.0,
        )
    ]
    # Patient needs 35 tablets. Cut strip = 35 * 5.0 = 175.0 BDT, waste = 0.
    plan = solve_divisible_solid_dispensing(35.0, skus, unit_name="tablet")
    assert plan.total_cost_bdt == 175.0
    assert plan.total_units_supplied == 35.0
    assert plan.leftover_waste == 0.0
    assert plan.strategy == "CUT_STRIP"


def test_divisible_solid_pack_discount_selection():
    """Verify that if buying full pack is cheaper than cut strip, full pack is chosen."""
    skus = [
        PackSKU(
            sku_index=0,
            unit_price_bdt=10.0,
            pack_qty=50,
            pack_price_bdt=300.0,  # 6.0 BDT per unit when buying full pack
            effective_per_unit_bdt=10.0,
        )
    ]
    # Patient needs 40 tablets.
    # Cut strip: 40 * 10 = 400.0 BDT
    # Full pack of 50 tablets: 300.0 BDT -> Buying full pack saves 100 BDT!
    plan = solve_divisible_solid_dispensing(40.0, skus, unit_name="tablet")
    assert plan.total_cost_bdt == 300.0
    assert plan.total_units_supplied == 50.0
    assert plan.leftover_waste == 10.0
    assert plan.strategy == "FULL_PACK"


def test_regimen_polypharmacy_duplicate_active_ingredient_alert():
    """Verify polypharmacy safety audit flags duplicate active ingredients across different brand names."""
    mock_df = pl.DataFrame(
        {
            "brand id": ["1", "2"],
            "brand name": ["Napa", "Ace"],
            "type": ["allopathic", "allopathic"],
            "slug": ["napa-500", "ace-500"],
            "dosage form": ["Tablet", "Tablet"],
            "generic": ["Paracetamol", "Paracetamol"],
            "strength": ["500 mg", "500 mg"],
            "manufacturer": ["Beximco", "Square"],
            "package container": ["Unit Price: ৳ 1.20", "Unit Price: ৳ 1.20"],
            "Package Size": [None, None],
        }
    )
    optimizer = RegimenOptimizer(catalog_df=mock_df)

    prescription = [
        PrescriptionItem(brand_id="1", units_per_dose=1.0, doses_per_day=3.0, duration_days=5),
        PrescriptionItem(brand_id="2", units_per_dose=1.0, doses_per_day=2.0, duration_days=5),
    ]

    res = optimizer.optimize_regimen(prescription)
    assert res.has_duplicate_ingredients is True
    dup_alerts = [a for a in res.safety_alerts if a.alert_type == "DUPLICATE_ACTIVE_INGREDIENT"]
    assert len(dup_alerts) == 1
    assert "paracetamol" in dup_alerts[0].active_ingredient.lower()
    assert set(dup_alerts[0].affected_brands) == {"Napa", "Ace"}


def test_regimen_nti_brand_locked_from_substitution():
    """Verify that Narrow Therapeutic Index drugs are strictly locked from brand substitution."""
    mock_df = pl.DataFrame(
        {
            "brand id": ["101", "102"],
            "brand name": ["Unifarin", "CheapFarin"],
            "type": ["allopathic", "allopathic"],
            "slug": ["unifarin-5", "cheapfarin-5"],
            "dosage form": ["Tablet", "Tablet"],
            "generic": ["Warfarin Sodium", "Warfarin Sodium"],
            "strength": ["5 mg", "5 mg"],
            "manufacturer": ["Delta", "GenericPharma"],
            "package container": ["Unit Price: ৳ 5.00", "Unit Price: ৳ 1.00"],
            "Package Size": [None, None],
        }
    )
    optimizer = RegimenOptimizer(catalog_df=mock_df)

    prescription = [
        PrescriptionItem(brand_id="101", units_per_dose=1.0, doses_per_day=1.0, duration_days=30),
    ]

    res = optimizer.optimize_regimen(prescription)
    assert res.has_nti_lock is True
    item_res = res.items[0]
    assert item_res.is_nti_locked is True
    assert item_res.is_substitutable is False
    assert "Narrow Therapeutic Index" in item_res.substitution_block_reason
    assert item_res.optimal_brand_id == "101"
    assert item_res.optimal_brand_name == "Unifarin"
    assert item_res.savings_bdt == 0.0


def test_regimen_end_to_end_substitution_and_savings():
    """Verify that safe allopathic drugs are substituted with cheaper bioequivalent alternatives."""
    mock_df = pl.DataFrame(
        {
            "brand id": ["201", "202"],
            "brand name": ["ExpensiveCipro", "AffordableCipro"],
            "type": ["allopathic", "allopathic"],
            "slug": ["exp-cipro-500", "aff-cipro-500"],
            "dosage form": ["Tablet", "Tablet"],
            "generic": ["Ciprofloxacin", "Ciprofloxacin"],
            "strength": ["500 mg", "500 mg"],
            "manufacturer": ["PharmaCorp", "ValuePharma"],
            "package container": ["Unit Price: ৳ 15.00", "Unit Price: ৳ 8.00"],
            "Package Size": [None, None],
        }
    )
    optimizer = RegimenOptimizer(catalog_df=mock_df)

    prescription = [
        PrescriptionItem(brand_id="201", units_per_dose=1.0, doses_per_day=2.0, duration_days=10),
    ]

    res = optimizer.optimize_regimen(prescription)
    assert res.has_duplicate_ingredients is False
    assert res.has_nti_lock is False
    item_res = res.items[0]
    assert item_res.is_substitutable is True
    assert item_res.prescribed_cost_bdt == 300.0  # 20 tabs * 15 BDT
    assert item_res.optimal_brand_name == "AffordableCipro"
    assert item_res.optimal_cost_bdt == 160.0    # 20 tabs * 8 BDT
    assert item_res.savings_bdt == 140.0
    assert item_res.savings_pct == 46.67
    assert res.total_savings_bdt == 140.0
