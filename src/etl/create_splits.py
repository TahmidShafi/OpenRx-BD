"""Partition raw dataset into three immutable splits with strict anti-leakage firewall."""

from collections import Counter
from pathlib import Path
import random
import polars as pl
from src.etl.split_guard import record_split_lock
from src.profiler.template_profiler import normalize_sku_chunk

SEED = 20261016
SPLITS_DIR = Path("data/splits")
RAW_PARQUET_PATH = Path("data/raw/medicines.parquet")

UNSEEN_HOLDOUT_PATH = SPLITS_DIR / "unseen_template_holdout.parquet"
DEV_SPLIT_PATH = SPLITS_DIR / "dev_split.parquet"
BLIND_TEST_PATH = SPLITS_DIR / "blind_test.parquet"

EXPECTED_COLUMNS = [
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


def create_splits(
    raw_path: Path = RAW_PARQUET_PATH,
    splits_dir: Path = SPLITS_DIR,
    seed: int = SEED,
) -> dict[str, int]:
    """Partition raw medicines dataset into three immutable partitions:

    1. unseen_template_holdout.parquet: Templates ranked #16 through #35.
    2. dev_split.parquet: Stratified 70% of remaining rows by dosage form & template ID.
    3. blind_test.parquet: Stratified 30% of remaining rows (locked evaluation set).
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {raw_path}. Run fetch_dataset first.")

    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_parquet(raw_path)
    # Filter for non-null package container rows
    df_valid = df.filter(
        pl.col("package container").is_not_null()
        & (pl.col("package container").str.strip_chars() != "")
    )
    total_valid = len(df_valid)

    # 1. Compute structural template for each row
    row_templates: list[str] = []
    for raw in df_valid["package container"].to_list():
        chunks = [c.strip() for c in str(raw).split(",,") if c.strip()]
        norm_chunks = [normalize_sku_chunk(c) for c in chunks]
        row_templates.append(",,".join(norm_chunks))

    # Rank templates by frequency descending
    counts = Counter(row_templates)
    ranked_templates = [t for t, _ in counts.most_common()]
    template_to_rank = {t: idx + 1 for idx, t in enumerate(ranked_templates)}

    df_valid = df_valid.with_columns(
        [
            pl.Series("template_rank", [template_to_rank[t] for t in row_templates]),
        ]
    )

    # 2. Partition 1: Unseen template holdout (ranked #16 through #35 inclusive)
    holdout_df = df_valid.filter(
        (pl.col("template_rank") >= 16) & (pl.col("template_rank") <= 35)
    ).select(EXPECTED_COLUMNS)

    remaining_df = df_valid.filter(
        (pl.col("template_rank") < 16) | (pl.col("template_rank") > 35)
    )

    # 3. Stratified 70/30 split on remaining rows by (dosage form, template_rank)
    rng = random.Random(seed)
    groups: dict[tuple[str, int], list[int]] = {}
    for idx, row in enumerate(remaining_df.iter_rows(named=True)):
        stratum = (str(row["dosage form"]).strip().lower(), row["template_rank"])
        groups.setdefault(stratum, []).append(idx)

    dev_indices: list[int] = []
    blind_indices: list[int] = []

    for stratum, idx_list in sorted(groups.items(), key=lambda x: str(x[0])):
        rng.shuffle(idx_list)
        n_total = len(idx_list)
        n_dev = int(round(n_total * 0.7))
        dev_indices.extend(idx_list[:n_dev])
        blind_indices.extend(idx_list[n_dev:])

    dev_df = remaining_df[dev_indices].select(EXPECTED_COLUMNS)
    blind_df = remaining_df[blind_indices].select(EXPECTED_COLUMNS)

    # Save to parquet
    holdout_df.write_parquet(UNSEEN_HOLDOUT_PATH, compression="snappy")
    dev_df.write_parquet(DEV_SPLIT_PATH, compression="snappy")
    blind_df.write_parquet(BLIND_TEST_PATH, compression="snappy")

    summary_stats = {
        "total_non_null_rows": total_valid,
        "unseen_template_holdout_rows": len(holdout_df),
        "dev_split_rows": len(dev_df),
        "blind_test_rows": len(blind_df),
        "holdout_pct": round(len(holdout_df) / total_valid * 100, 2),
        "dev_pct_of_remaining": round(len(dev_df) / len(remaining_df) * 100, 2),
        "blind_pct_of_remaining": round(len(blind_df) / len(remaining_df) * 100, 2),
    }

    # Record cryptographic lock manifest for blind_test.parquet
    record_split_lock(BLIND_TEST_PATH, seed=seed, summary_stats=summary_stats)

    print("\n[Splits Generated Successfully]")
    print(f"  Total Non-Null Rows         : {total_valid:,}")
    print(f"  1. Unseen Template Holdout  : {len(holdout_df):,} rows -> {UNSEEN_HOLDOUT_PATH}")
    print(f"  2. Dev Split (70%)          : {len(dev_df):,} rows -> {DEV_SPLIT_PATH}")
    print(f"  3. Blind Test Split (30%)   : {len(blind_df):,} rows -> {BLIND_TEST_PATH}")
    print(f"  Checksum lock recorded at   : reports/split_lock.json\n")

    return summary_stats


if __name__ == "__main__":
    create_splits()
