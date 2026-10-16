"""Dataset ingestion and schema validation for OpenRx-BD."""

import argparse
from pathlib import Path
import sys
import urllib.request
import pandas as pd
from openrx_bd.schema import GROUND_TRUTH_COLUMNS, validate_dataframe_schema

HF_MEDICINE_CSV_URL = (
    "https://huggingface.co/datasets/Mahadih534/all-Bangladeshi-medicines/raw/main/medicine.csv"
)
DEFAULT_RAW_PATH = Path("data/raw/medicine.csv")


def download_raw_dataset(
    url: str = HF_MEDICINE_CSV_URL,
    dest_path: Path = DEFAULT_RAW_PATH,
    force: bool = False,
) -> Path:
    """Download the raw Hugging Face CSV if not already cached in data/raw/."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and not force:
        print(f"[Ingest] Found existing cached dataset at: {dest_path}")
        return dest_path

    print(f"[Ingest] Downloading upstream dataset from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "OpenRx-BD/0.1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    with open(dest_path, "wb") as f:
        f.write(content)

    print(f"[Ingest] Successfully downloaded {len(content):,} bytes to {dest_path}")
    return dest_path


def load_raw_dataset(csv_path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load raw dataset with strict UTF-8 encoding and schema validation."""
    if not csv_path.exists():
        download_raw_dataset(dest_path=csv_path)

    df = pd.read_csv(csv_path, dtype=str, encoding="utf-8")
    validate_dataframe_schema(df)
    return df


def verify_schema(csv_path: Path = DEFAULT_RAW_PATH) -> bool:
    """Validate that the dataset matches the exact 10-column schema."""
    df = load_raw_dataset(csv_path)
    print(f"[Schema Check] PASSED: Dataset matches exact 10-column schema.")
    print(f"[Schema Check] Total Rows: {len(df):,}")
    print(f"[Schema Check] Unique Generics: {df['generic'].nunique():,}")
    print(f"[Schema Check] Unique Dosage Forms: {df['dosage form'].nunique():,}")
    print(f"[Schema Check] Unique Manufacturers: {df['manufacturer'].nunique():,}")
    print(f"[Schema Check] Columns verified: {list(GROUND_TRUTH_COLUMNS)}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest and validate Bangladeshi medicines dataset.")
    parser.add_argument(
        "--verify-schema",
        action="store_true",
        help="Download (if needed) and strictly verify the 10-column schema.",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download raw dataset without schema verification.",
    )
    args = parser.parse_args()

    if args.download_only:
        download_raw_dataset()
        sys.exit(0)

    if args.verify_schema or len(sys.argv) == 1:
        success = verify_schema()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
