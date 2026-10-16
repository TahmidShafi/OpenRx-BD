"""Split guard: records cryptographic lock and checksums for the blind evaluation split."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import polars as pl

SPLIT_LOCK_PATH = Path("reports/split_lock.json")


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def record_split_lock(
    blind_test_path: Path,
    seed: int,
    summary_stats: dict | None = None,
    lock_path: Path = SPLIT_LOCK_PATH,
) -> dict:
    """Record SHA-256 checksum, row count, and provenance of blind_test.parquet."""
    if not blind_test_path.exists():
        raise FileNotFoundError(f"Missing blind test split at {blind_test_path}")

    # Compute row count without inspecting row content
    blind_df = pl.read_parquet(blind_test_path)
    row_count = len(blind_df)
    sha256 = compute_file_sha256(blind_test_path)

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_data = {
        "blind_test_file": str(blind_test_path).replace("\\", "/"),
        "sha256": sha256,
        "row_count": row_count,
        "seed": seed,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": summary_stats or {},
    }

    with open(lock_path, "w", encoding="utf-8") as f:
        json.dump(lock_data, f, indent=2)

    print(f"[Split Guard] Locked {blind_test_path} -> {lock_path} (SHA-256: {sha256[:16]}...)")
    return lock_data


def verify_split_lock(
    blind_test_path: Path,
    lock_path: Path = SPLIT_LOCK_PATH,
) -> bool:
    """Verify that blind_test.parquet has not been modified or tampered with."""
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing lock manifest at {lock_path}")
    if not blind_test_path.exists():
        raise FileNotFoundError(f"Missing blind test split at {blind_test_path}")

    with open(lock_path, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    current_sha256 = compute_file_sha256(blind_test_path)
    current_count = len(pl.read_parquet(blind_test_path))

    if current_sha256 != lock_data["sha256"]:
        raise ValueError(
            f"Blind test checksum mismatch! Expected: {lock_data['sha256']}, got: {current_sha256}"
        )

    if current_count != lock_data["row_count"]:
        raise ValueError(
            f"Blind test row count mismatch! Expected: {lock_data['row_count']}, got: {current_count}"
        )

    return True
