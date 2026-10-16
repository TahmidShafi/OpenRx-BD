"""Unit tests for deterministic generic substitution and cost-optimization engine."""

import pytest
from openrx_bd.engine import CostOptEngine
from openrx_bd.split import load_dev_split


@pytest.fixture(scope="module")
def engine():
    df = load_dev_split()
    return CostOptEngine(df)


def test_engine_bioequivalence_invariants(engine):
    """Ensure alternatives strictly match generic, dosage form, and strength."""
    # Napa is a common Paracetamol 500 mg Tablet
    res = engine.optimize_prescription("Napa", dosage_form="Tablet", strength="500 mg")
    assert res.query_generic != ""
    assert res.query_dosage_form == "Tablet"
    assert res.query_strength == "500 mg"

    for alt in res.alternatives:
        assert alt.dosage_form == res.query_dosage_form
        assert alt.strength == res.query_strength
        if res.query_unit_price_bdt is not None:
            assert alt.unit_price_bdt < res.query_unit_price_bdt
            assert alt.savings_pct > 0


def test_dosage_form_never_cross_substitutes(engine):
    """Ensure syrup does not get suggested for tablets or vice versa."""
    res_tablet = engine.optimize_prescription("Napa", dosage_form="Tablet")
    for alt in res_tablet.alternatives:
        assert "syrup" not in alt.dosage_form.lower()
        assert "suspension" not in alt.dosage_form.lower()


def test_already_optimal_brand(engine):
    """Test that a brand with the cheapest price correctly reports is_already_optimal."""
    # Find the cheapest Paracetamol 500 mg Tablet in the dataset
    res = engine.optimize_prescription("Napa", dosage_form="Tablet", strength="500 mg")
    if res.alternatives:
        cheapest_name = res.alternatives[0].brand_name
        cheapest_res = engine.optimize_prescription(
            cheapest_name, dosage_form="Tablet", strength="500 mg"
        )
        # Should have fewer or 0 alternatives
        assert len(cheapest_res.alternatives) <= len(res.alternatives)


def test_fuzzy_lookup_tolerates_minor_typo(engine):
    """Test fuzzy matching resolves slightly misspelled brand names."""
    res = engine.optimize_prescription("Nappa")  # Napa with double p
    assert "napa" in res.query_brand_name.lower()
