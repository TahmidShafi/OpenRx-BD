"""Deterministic, leakage-proof dataset splitting with anti-contamination firewall."""

import hashlib
from pathlib import Path
import pandas as pd
from openrx_bd.ingest import load_raw_dataset
from openrx_bd.schema import validate_dataframe_schema

DEFAULT_SPLIT_DIR = Path("data/splits")
DEV_SPLIT_PATH = DEFAULT_SPLIT_DIR / "dev_split.parquet"
BLIND_TEST_SPLIT_PATH = DEFAULT_SPLIT_DIR / "blind_test.parquet"


class DataLeakageFirewallError(PermissionError):
    """Raised when an agent or unauthorized module attempts to inspect blind_test.parquet."""


def hash_split_key(key: str, salt: str = "openrx_bd_v1") -> int:
    """Deterministic hash returning integer 0-99."""
    h = hashlib.sha256(f"{salt}_{key}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def create_deterministic_splits(
    df: pd.DataFrame | None = None,
    dev_ratio: float = 0.80,
    dest_dir: Path = DEFAULT_SPLIT_DIR,
) -> tuple[Path, Path]:
    """Split raw dataset into dev_split (80%) and blind_test (20%) using deterministic hashing.

    The split is stratified on 'generic' + 'brand id' so that splits are repeatable across runs.
    """
    if df is None:
        df = load_raw_dataset()

    validate_dataframe_schema(df)

    threshold = int(dev_ratio * 100)
    dev_mask = df.apply(
        lambda row: hash_split_key(f"{row['generic']}::{row['brand id']}") < threshold,
        axis=1,
    )

    dev_df = df[dev_mask].reset_index(drop=True)
    blind_df = df[~dev_mask].reset_index(drop=True)

    dest_dir.mkdir(parents=True, exist_ok=True)
    dev_path = dest_dir / "dev_split.parquet"
    blind_path = dest_dir / "blind_test.parquet"

    dev_df.to_parquet(dev_path, engine="pyarrow", index=False)
    blind_df.to_parquet(blind_path, engine="pyarrow", index=False)

    print(f"[Split] Dev set saved to {dev_path} ({len(dev_df):,} rows)")
    print(f"[Split] Blind test set saved to {blind_path} ({len(blind_df):,} rows)")

    return dev_path, blind_path


def load_dev_split(path: Path = DEV_SPLIT_PATH) -> pd.DataFrame:
    """Safe loader for development and parser tuning."""
    if not path.exists():
        create_deterministic_splits()
    return pd.read_parquet(path)


def load_blind_test_split(
    path: Path = BLIND_TEST_SPLIT_PATH,
    authorized_for_final_eval: bool = False,
) -> pd.DataFrame:
    """Firewalled loader for blind_test.parquet.

    Raises DataLeakageFirewallError if called during normal development or parser tuning.
    """
    if not authorized_for_final_eval:
        raise DataLeakageFirewallError(
            "VIOLATION OF RULE 2: Agent is strictly forbidden from reading, inspecting, "
            "or sampling from blind_test.parquet during parser or engine development! "
            "Use load_dev_split() instead."
        )

    if not path.exists():
        create_deterministic_splits()
    return pd.read_parquet(path)


if __name__ == "__main__":
    create_deterministic_splits()
