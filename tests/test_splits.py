"""Unit tests verifying dataset partition integrity and zero data leakage."""

from pathlib import Path
import json
import polars as pl
import pytest
from src.etl.create_splits import (
    BLIND_TEST_PATH,
    DEV_SPLIT_PATH,
    EXPECTED_COLUMNS,
    RAW_PARQUET_PATH,
    UNSEEN_HOLDOUT_PATH,
    create_splits,
)
from src.etl.split_guard import SPLIT_LOCK_PATH, verify_split_lock


@pytest.fixture(scope="module")
def ensure_splits_exist():
    """Ensure splits and split lock exist before running tests."""
    if (
        not UNSEEN_HOLDOUT_PATH.exists()
        or not DEV_SPLIT_PATH.exists()
        or not BLIND_TEST_PATH.exists()
        or not SPLIT_LOCK_PATH.exists()
    ):
        create_splits()


def test_splits_files_exist_and_match_10_columns(ensure_splits_exist):
    """Verify that all three split partitions exist and maintain the exact 10-column schema."""
    for split_path in [UNSEEN_HOLDOUT_PATH, DEV_SPLIT_PATH, BLIND_TEST_PATH]:
        assert split_path.exists(), f"Split file {split_path} does not exist"
        # Only inspect metadata schema, never content rows of blind/holdout sets
        df = pl.read_parquet(split_path)
        assert df.columns == EXPECTED_COLUMNS, (
            f"Schema mismatch in {split_path}! Expected {EXPECTED_COLUMNS}, got {df.columns}"
        )


def test_zero_overlap_of_brand_ids(ensure_splits_exist):
    """Verify strictly zero overlap of brand id across dev, blind, and holdout splits."""
    dev_ids = set(pl.read_parquet(DEV_SPLIT_PATH).select("brand id")["brand id"].to_list())
    blind_ids = set(pl.read_parquet(BLIND_TEST_PATH).select("brand id")["brand id"].to_list())
    holdout_ids = set(
        pl.read_parquet(UNSEEN_HOLDOUT_PATH).select("brand id")["brand id"].to_list()
    )

    # Check pairwise disjointness
    dev_blind_overlap = dev_ids.intersection(blind_ids)
    assert len(dev_blind_overlap) == 0, f"Data leakage between dev and blind! Overlap: {len(dev_blind_overlap)}"

    dev_holdout_overlap = dev_ids.intersection(holdout_ids)
    assert len(dev_holdout_overlap) == 0, f"Data leakage between dev and holdout! Overlap: {len(dev_holdout_overlap)}"

    blind_holdout_overlap = blind_ids.intersection(holdout_ids)
    assert len(blind_holdout_overlap) == 0, f"Data leakage between blind and holdout! Overlap: {len(blind_holdout_overlap)}"


def test_total_row_sum_equals_non_null_dataset_rows(ensure_splits_exist):
    """Verify that the sum of rows across all 3 splits exactly equals total non-null dataset rows."""
    raw_df = pl.read_parquet(RAW_PARQUET_PATH)
    valid_raw_count = len(
        raw_df.filter(
            pl.col("package container").is_not_null()
            & (pl.col("package container").str.strip_chars() != "")
        )
    )

    dev_count = len(pl.read_parquet(DEV_SPLIT_PATH))
    blind_count = len(pl.read_parquet(BLIND_TEST_PATH))
    holdout_count = len(pl.read_parquet(UNSEEN_HOLDOUT_PATH))

    total_split_rows = dev_count + blind_count + holdout_count
    assert total_split_rows == valid_raw_count, (
        f"Partition row sum mismatch! Splits total: {total_split_rows:,}, Non-null raw: {valid_raw_count:,}"
    )


def test_split_lock_integrity(ensure_splits_exist):
    """Verify that reports/split_lock.json exists and accurately locks blind_test.parquet."""
    assert SPLIT_LOCK_PATH.exists(), f"Missing split lock at {SPLIT_LOCK_PATH}"

    # Verify lock checksum and row count
    assert verify_split_lock(BLIND_TEST_PATH, SPLIT_LOCK_PATH) is True

    with open(SPLIT_LOCK_PATH, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    assert lock_data["seed"] == 20261016
    assert lock_data["row_count"] > 0
    assert len(lock_data["sha256"]) == 64
