"""Unit tests for the non-silent Pydantic package container parser."""

import pytest
from openrx_bd.parser import (
    InvariantViolation,
    ParsedAbstain,
    ParsedSuccess,
    parse_package_container,
)
from openrx_bd.split import load_dev_split


def test_parse_unit_and_pack():
    """Verify standard MedEx unit + pack pricing format."""
    raw = "Unit Price: ৳ 5.98,(100's pack: ৳ 598.00),"
    result = parse_package_container(raw)
    assert isinstance(result, ParsedSuccess)
    assert result.status == "success"
    assert result.unit_price_bdt == 5.98
    assert result.pack_size == 100
    assert result.pack_price_bdt == 598.00
    assert result.container_type == "unit"


def test_parse_vial_container_and_pack():
    """Verify vial container with packaging format."""
    raw = "500 mg vial: ৳ 28.43,(5's pack: ৳ 142.15),"
    result = parse_package_container(raw)
    assert isinstance(result, ParsedSuccess)
    assert result.status == "success"
    assert result.unit_price_bdt == 28.43
    assert result.pack_size == 5
    assert result.pack_price_bdt == 142.15
    assert "vial" in result.container_type.lower()


def test_parse_single_bottle():
    """Verify bottle container with single price."""
    raw = "100 ml bottle: ৳ 70.00"
    result = parse_package_container(raw)
    assert isinstance(result, ParsedSuccess)
    assert result.status == "success"
    assert result.unit_price_bdt == 70.00
    assert result.pack_size == 1
    assert result.container_type == "100 ml bottle"


def test_parse_abstain_on_null_and_empty():
    """Verify explicit ParsedAbstain on null or whitespace."""
    assert isinstance(parse_package_container(None), ParsedAbstain)
    assert isinstance(parse_package_container("   "), ParsedAbstain)


def test_parse_abstain_on_unavailable():
    """Verify explicit ParsedAbstain on 'Price Unavailable' or 'Not for sale'."""
    res1 = parse_package_container("Price Unavailable")
    assert isinstance(res1, ParsedAbstain)
    assert res1.status == "abstain"

    res2 = parse_package_container("Not for sale")
    assert isinstance(res2, ParsedAbstain)
    assert res2.status == "abstain"


def test_parse_invariant_violation_on_math_mismatch():
    """Verify InvariantViolation when pack price completely contradicts unit price * pack_size."""
    # 5.00 * 100 should be 500, but claims 9999.00
    bogus_text = "Unit Price: ৳ 5.00,(100's pack: ৳ 9999.00),"
    result = parse_package_container(bogus_text)
    assert isinstance(result, InvariantViolation)
    assert result.status == "invariant_violation"
    assert "Price mismatch" in result.violation


def test_parse_dev_split_zero_crashes():
    """Ensure parser runs over thousands of dev_split rows with 0 unhandled exceptions."""
    dev_df = load_dev_split()
    results = [parse_package_container(x) for x in dev_df["package container"]]

    successes = [r for r in results if isinstance(r, ParsedSuccess)]
    abstains = [r for r in results if isinstance(r, ParsedAbstain)]
    violations = [r for r in results if isinstance(r, InvariantViolation)]

    assert len(results) == len(dev_df)
    # The vast majority of records in dev_split must parse successfully (>95%)
    success_rate = len(successes) / len(results)
    assert success_rate > 0.95, f"Success rate {success_rate:.2%} lower than 95%"
    print(f"\n[Parser Audit] Successes: {len(successes):,} ({success_rate:.2%}), "
          f"Abstains: {len(abstains):,}, Violations: {len(violations):,}")
