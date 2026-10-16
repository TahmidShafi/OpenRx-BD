"""ETL script to dynamically fetch, validate, and hash the Bangladeshi medicines dataset."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import polars as pl

# Exact 10 columns as strictly defined in AGENT.md
EXPECTED_COLUMNS: list[str] = [
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
]

RAW_DIR = Path("data/raw")
RAW_PARQUET_PATH = RAW_DIR / "medicines.parquet"
MANIFEST_PATH = RAW_DIR / "manifest.json"
HF_DATASET_NAME = "Mahadih534/all-Bangladeshi-medicines"
HF_DIRECT_CSV_URL = (
    "https://huggingface.co/datasets/Mahadih534/all-Bangladeshi-medicines/raw/main/medicine.csv"
)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_schema(df: pl.DataFrame) -> None:
    """Assert that the exact 10 columns from AGENT.md are present."""
    actual_columns = df.columns
    if actual_columns != EXPECTED_COLUMNS:
        missing = set(EXPECTED_COLUMNS) - set(actual_columns)
        unexpected = set(actual_columns) - set(EXPECTED_COLUMNS)
        err = f"Schema validation failed! Missing: {missing}, Unexpected: {unexpected}"
        raise ValueError(err)
    print(f"[Schema Lock] Verified exact 10 columns: {EXPECTED_COLUMNS}")


def fetch_dataset() -> pl.DataFrame:
    """Fetch dataset using Hugging Face datasets with fallback to direct HF Hub CSV."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ETL] Fetching dataset '{HF_DATASET_NAME}' ...")

    df: pl.DataFrame | None = None

    # Method 1: Hugging Face datasets library
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET_NAME)
        table = ds["train"].data.table
        df = pl.from_arrow(table)
        print(f"[ETL] Successfully loaded via huggingface datasets library ({len(df):,} rows)")
    except Exception as exc:
        print(f"[ETL] datasets.load_dataset encountered error: {exc}. Trying direct download fallback...")

    # Method 2: Direct CSV download fallback
    if df is None:
        import urllib.request
        temp_csv = RAW_DIR / "temp_medicine.csv"
        req = urllib.request.Request(HF_DIRECT_CSV_URL, headers={"User-Agent": "OpenRx-BD/0.1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(temp_csv, "wb") as f:
            f.write(resp.read())
        df = pl.read_csv(temp_csv, infer_schema_length=0)
        if temp_csv.exists():
            temp_csv.unlink()
        print(f"[ETL] Successfully loaded via direct CSV download ({len(df):,} rows)")

    # Ensure all columns are strings and match exact names
    # Ensure columns match order
    df = df.select([pl.col(c).cast(pl.Utf8) for c in EXPECTED_COLUMNS])
    verify_schema(df)

    # Save to data/raw/medicines.parquet
    df.write_parquet(RAW_PARQUET_PATH, compression="snappy")
    print(f"[ETL] Saved raw dataset to {RAW_PARQUET_PATH}")

    # Compute and record SHA-256 in manifest.json
    file_sha256 = compute_sha256(RAW_PARQUET_PATH)
    manifest_data = {
        "dataset_name": HF_DATASET_NAME,
        "raw_file": str(RAW_PARQUET_PATH).replace("\\", "/"),
        "sha256": file_sha256,
        "row_count": len(df),
        "column_count": len(EXPECTED_COLUMNS),
        "columns": EXPECTED_COLUMNS,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"[ETL] Manifest saved to {MANIFEST_PATH} (SHA-256: {file_sha256[:16]}...)")
    return df


if __name__ == "__main__":
    fetch_dataset()
