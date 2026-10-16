"""Blind-test evaluation gate for evaluating parser and engine on held-out split."""

import pytest
from openrx_bd.engine import CostOptEngine
from openrx_bd.parser import ParsedSuccess, parse_package_container
from openrx_bd.split import BLIND_TEST_SPLIT_PATH, load_blind_test_split


def test_blind_test_evaluation_yield():
    """Evaluate parser generalization on held-out blind_test.parquet."""
    assert BLIND_TEST_SPLIT_PATH.exists(), "blind_test.parquet must exist"

    # Authorized loader explicitly for the final gate
    blind_df = load_blind_test_split(authorized_for_final_eval=True)

    results = [parse_package_container(x) for x in blind_df["package container"]]
    successes = [r for r in results if isinstance(r, ParsedSuccess)]

    yield_rate = len(successes) / len(results)
    print(f"\n[Blind Test Evaluation] Total: {len(results):,}, Parsed: {len(successes):,} ({yield_rate:.2%})")

    # The parser must generalize with > 95% yield on unseen data
    assert yield_rate >= 0.95, f"Yield {yield_rate:.2%} is below the 95% generalization target."


def test_blind_test_engine_optimization():
    """Verify that CostOptEngine can optimize on the blind test split without errors."""
    blind_df = load_blind_test_split(authorized_for_final_eval=True)
    engine = CostOptEngine(blind_df)

    # Pick a frequent brand in the blind test set
    frequent_brand = blind_df["brand name"].value_counts().index[0]
    res = engine.optimize_prescription(frequent_brand)

    assert res.query_generic != ""
    for alt in res.alternatives:
        assert alt.unit_price_bdt <= res.query_unit_price_bdt
