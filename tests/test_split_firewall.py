"""Unit tests for data leakage firewall and split integrity."""

from pathlib import Path
import pytest
from openrx_bd.schema import validate_dataframe_schema
from openrx_bd.split import (
    BLIND_TEST_SPLIT_PATH,
    DEV_SPLIT_PATH,
    DataLeakageFirewallError,
    create_deterministic_splits,
    load_blind_test_split,
    load_dev_split,
)


def test_split_generation_and_disjointness():
    """Verify that train/blind splits are generated and have zero record leakage."""
    dev_path, blind_path = create_deterministic_splits()

    assert dev_path.exists()
    assert blind_path.exists()

    dev_df = load_dev_split(dev_path)
    # Authorized loader only for the firewall test verification
    blind_df = load_blind_test_split(blind_path, authorized_for_final_eval=True)

    # 1. Check schemas
    validate_dataframe_schema(dev_df)
    validate_dataframe_schema(blind_df)

    # 2. Check disjointness (zero leakage between splits)
    dev_ids = set(dev_df["brand id"].astype(str))
    blind_ids = set(blind_df["brand id"].astype(str))
    overlap = dev_ids.intersection(blind_ids)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping brand IDs: {overlap}"

    # 3. Check proportions (~80/20)
    total = len(dev_df) + len(blind_df)
    dev_pct = len(dev_df) / total
    assert 0.75 <= dev_pct <= 0.85, f"Unexpected split ratio: {dev_pct:.2f}"


def test_firewall_blocks_unauthorized_read():
    """Verify that unauthorized inspection of blind_test.parquet is actively blocked."""
    with pytest.raises(DataLeakageFirewallError, match="VIOLATION OF RULE 2"):
        load_blind_test_split(authorized_for_final_eval=False)
