"""Independent blind benchmark evaluation and market price dispersion audit."""

from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import numpy as np
import polars as pl

from src.ontology.entity_resolver import resolve_medicine_entity
from src.parser.container_parser import parse_dataframe
from src.parser.schemas import RowParseResult


# Ensure stdout prints properly across all terminals and platforms
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


SPLIT_LOCK_PATH = Path("reports/split_lock.json")
BLIND_TEST_PATH = Path("data/splits/blind_test.parquet")
DEV_SPLIT_PATH = Path("data/splits/dev_split.parquet")
UNSEEN_HOLDOUT_PATH = Path("data/splits/unseen_template_holdout.parquet")
RAW_DATASET_PATH = Path("data/raw/medicines.parquet")
OUTPUT_BENCHMARK_PATH = Path("reports/benchmark_results.json")


def verify_blind_test_lock(
    blind_path: Path = BLIND_TEST_PATH,
    lock_path: Path = SPLIT_LOCK_PATH,
) -> str:
    """Verify cryptographic SHA-256 integrity of blind_test.parquet against split_lock.json."""
    if not lock_path.exists():
        raise FileNotFoundError(f"Split lock manifest not found at {lock_path}")
    if not blind_path.exists():
        raise FileNotFoundError(f"Blind test dataset not found at {blind_path}")

    with open(lock_path, "r", encoding="utf-8") as f:
        lock_data = json.load(f)

    expected_sha = lock_data["sha256"]

    h = hashlib.sha256()
    with open(blind_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest()

    if actual_sha != expected_sha:
        raise RuntimeError(
            f"CRYPTOGRAPHIC INTEGRITY VIOLATION: blind_test.parquet hash mismatch!\n"
            f"Expected from lock : {expected_sha}\n"
            f"Actual computed     : {actual_sha}\n"
            f"The blind evaluation set was modified or contaminated!"
        )

    return actual_sha


def evaluate_split(split_name: str, split_path: Path) -> dict[str, Any]:
    """Execute frozen parser on a dataset split and calculate coverage and invariant metrics."""
    if not split_path.exists():
        raise FileNotFoundError(f"Split dataset {split_name} not found at {split_path}")

    df = pl.read_parquet(split_path)
    results: list[RowParseResult] = parse_dataframe(df)

    total_rows = len(results)
    valid_count = sum(1 for r in results if r.status == "PARSED_VALID")
    invariant_violation_count = sum(1 for r in results if r.status == "INVARIANT_VIOLATION")
    abstain_template_count = sum(
        1 for r in results if r.status == "ABSTAIN_UNSUPPORTED_TEMPLATE"
    )
    abstain_null_count = sum(1 for r in results if r.status == "ABSTAIN_NULL_CONTAINER")
    total_abstain_count = abstain_template_count + abstain_null_count

    coverage_pct = round((valid_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    abstention_pct = round((total_abstain_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    violation_pct = (
        round((invariant_violation_count / total_rows * 100), 2) if total_rows > 0 else 0.0
    )

    # SKU-level invariant audit
    all_skus = [s for r in results for s in r.skus]
    total_skus = len(all_skus)
    checked_skus = [s for s in all_skus if s.invariant_checked]
    checked_sku_count = len(checked_skus)
    passed_skus = [s for s in checked_skus if s.invariant_passed is True]
    passed_sku_count = len(passed_skus)

    sku_invariant_check_rate = (
        round((checked_sku_count / total_skus * 100), 2) if total_skus > 0 else 0.0
    )
    sku_invariant_pass_rate = (
        round((passed_sku_count / checked_sku_count * 100), 2) if checked_sku_count > 0 else 0.0
    )

    # Row-level invariant audit
    rows_with_check = sum(1 for r in results if any(s.invariant_checked for s in r.skus))
    row_check_rate = round((rows_with_check / total_rows * 100), 2) if total_rows > 0 else 0.0

    return {
        "split_name": split_name,
        "split_path": str(split_path),
        "total_rows": total_rows,
        "parsed_valid": valid_count,
        "coverage_pct": coverage_pct,
        "invariant_violations": invariant_violation_count,
        "invariant_violation_pct": violation_pct,
        "abstain_total": total_abstain_count,
        "abstain_pct": abstention_pct,
        "abstain_breakdown": {
            "unsupported_template": abstain_template_count,
            "null_or_empty_container": abstain_null_count,
        },
        "total_skus_parsed": total_skus,
        "skus_invariant_checked": checked_sku_count,
        "skus_invariant_passed": passed_sku_count,
        "invariant_check_rate_pct": sku_invariant_check_rate,
        "row_invariant_check_rate_pct": row_check_rate,
        "invariant_pass_rate_pct": sku_invariant_pass_rate,
    }


def compute_within_group_price_dispersion(
    df: pl.DataFrame,
    parsed_results: list[RowParseResult] | None = None,
) -> dict[str, Any]:
    """Compute within-equivalence-group price dispersion across the pharmaceutical catalog."""
    if parsed_results is None:
        parsed_results = parse_dataframe(df)

    equiv_groups: dict[Any, list[float]] = defaultdict(list)

    for row, res in zip(df.iter_rows(named=True), parsed_results):
        if res.status != "PARSED_VALID" or not res.skus:
            continue

        entity = resolve_medicine_entity(
            brand_id=row["brand id"],
            brand_name=row["brand name"],
            generic=row["generic"],
            dosage_form=row["dosage form"],
            strength=row.get("strength"),
            medicine_type=row.get("type", "allopathic"),
        )

        if not entity.equivalence_key or not entity.is_substitutable:
            continue

        valid_prices = [
            s.effective_per_unit_bdt for s in res.skus if s.effective_per_unit_bdt > 0
        ]
        if not valid_prices:
            continue

        min_brand_price = min(valid_prices)
        equiv_groups[entity.equivalence_key].append(min_brand_price)

    ratios: list[float] = []
    groups_gt_25_pct = 0

    for key, prices in equiv_groups.items():
        if len(prices) >= 2:
            p_min = min(prices)
            p_max = max(prices)
            if p_min > 0:
                ratio = p_max / p_min
                ratios.append(ratio)
                spread = (p_max - p_min) / p_min
                if spread > 0.25:
                    groups_gt_25_pct += 1

    total_equiv_groups = len(equiv_groups)
    multi_brand_groups = len(ratios)

    if ratios:
        q25, median, q75 = np.percentile(ratios, [25, 50, 75])
        iqr = q75 - q25
        pct_spread_gt_25 = round((groups_gt_25_pct / multi_brand_groups * 100), 2)
    else:
        q25, median, q75, iqr = 1.0, 1.0, 1.0, 0.0
        pct_spread_gt_25 = 0.0

    return {
        "total_equivalence_groups_indexed": total_equiv_groups,
        "multi_brand_groups_evaluated": multi_brand_groups,
        "median_max_to_min_price_ratio": round(float(median), 3),
        "q25_price_ratio": round(float(q25), 3),
        "q75_price_ratio": round(float(q75), 3),
        "iqr_price_ratio": round(float(iqr), 3),
        "groups_with_price_spread_gt_25_pct_count": groups_gt_25_pct,
        "groups_with_price_spread_gt_25_pct_rate": pct_spread_gt_25,
    }


def run_benchmark() -> dict[str, Any]:
    """Execute complete frozen blind benchmark across all three splits."""
    print("\n" + "=" * 78)
    print("OPENRX-BD FROZEN BENCHMARK & PRICE DISPERSION AUDIT")
    print("=" * 78)

    # 1. Cryptographic Firewall Verification
    sha256_lock = verify_blind_test_lock()
    print(f"[*] Cryptographic Lock Verified: blind_test.parquet is UNTOUCHED")
    print(f"    SHA-256: {sha256_lock}\n")

    # 2. Evaluate Splits
    print("[*] Running Frozen Parser on Dev Split...")
    dev_metrics = evaluate_split("dev_split", DEV_SPLIT_PATH)

    print("[*] Running Frozen Parser on Blind Test Split...")
    blind_metrics = evaluate_split("blind_test", BLIND_TEST_PATH)

    print("[*] Running Frozen Parser on Unseen Template Holdout Split...")
    holdout_metrics = evaluate_split("unseen_template_holdout", UNSEEN_HOLDOUT_PATH)

    # 3. Compute Price Dispersion Across the Complete Valid Dataset
    print("[*] Computing within-group pharmaceutical price dispersion across dataset...")
    full_df = pl.read_parquet(RAW_DATASET_PATH)
    dispersion_metrics = compute_within_group_price_dispersion(full_df)

    benchmark_payload = {
        "benchmark_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cryptographic_verification": {
            "blind_test_sha256": sha256_lock,
            "status": "VERIFIED_UNTOUCHED",
        },
        "splits": {
            "dev_split": dev_metrics,
            "blind_test": blind_metrics,
            "unseen_template_holdout": holdout_metrics,
        },
        "market_price_dispersion": dispersion_metrics,
    }

    # Save output to reports/benchmark_results.json
    OUTPUT_BENCHMARK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(benchmark_payload, f, indent=2)

    print(f"[+] Saved complete benchmark results -> {OUTPUT_BENCHMARK_PATH}\n")

    # 4. Print Full Summary Table
    print_benchmark_summary(dev_metrics, blind_metrics, holdout_metrics, dispersion_metrics)

    return benchmark_payload


def print_benchmark_summary(
    dev: dict[str, Any],
    blind: dict[str, Any],
    holdout: dict[str, Any],
    dispersion: dict[str, Any],
) -> None:
    """Print clean formatted Markdown benchmark comparison table."""
    print("-" * 78)
    print("PARSER GENERALIZATION BENCHMARK SUMMARY TABLE")
    print("-" * 78)
    header = (
        f"| {'Metric':<30} | {'Dev Split':<12} | {'Blind Test':<12} | {'Holdout (Tail)':<14} |"
    )
    print(header)
    print("|" + "-" * 32 + "|" + "-" * 14 + "|" + "-" * 14 + "|" + "-" * 16 + "|")

    rows = [
        ("Total Rows", f"{dev['total_rows']:,}", f"{blind['total_rows']:,}", f"{holdout['total_rows']:,}"),
        ("Coverage % (Parsed Valid)", f"{dev['coverage_pct']:.2f}%", f"{blind['coverage_pct']:.2f}%", f"{holdout['coverage_pct']:.2f}%"),
        ("Invariant Check Rate %", f"{dev['invariant_check_rate_pct']:.2f}%", f"{blind['invariant_check_rate_pct']:.2f}%", f"{holdout['invariant_check_rate_pct']:.2f}%"),
        ("Invariant Pass Rate %", f"{dev['invariant_pass_rate_pct']:.2f}%", f"{blind['invariant_pass_rate_pct']:.2f}%", f"{holdout['invariant_pass_rate_pct']:.2f}%"),
        ("Invariant Violation Rate %", f"{dev['invariant_violation_pct']:.2f}%", f"{blind['invariant_violation_pct']:.2f}%", f"{holdout['invariant_violation_pct']:.2f}%"),
        ("Abstention Rate %", f"{dev['abstain_pct']:.2f}%", f"{blind['abstain_pct']:.2f}%", f"{holdout['abstain_pct']:.2f}%"),
    ]

    for title, d_val, b_val, h_val in rows:
        print(f"| {title:<30} | {d_val:<12} | {b_val:<12} | {h_val:<14} |")

    print("-" * 78)
    print("\n" + "-" * 78)
    print("WITHIN-EQUIVALENCE-GROUP PRICE DISPERSION (Full Dataset)")
    print("-" * 78)
    print(f"  Total Equivalence Groups Indexed   : {dispersion['total_equivalence_groups_indexed']:,}")
    print(f"  Multi-Brand Groups Evaluated (>=2) : {dispersion['multi_brand_groups_evaluated']:,}")
    print(f"  Median Max-to-Min Price Ratio      : {dispersion['median_max_to_min_price_ratio']:.3f}x")
    print(f"  25th Percentile (Q1) Price Ratio   : {dispersion['q25_price_ratio']:.3f}x")
    print(f"  75th Percentile (Q3) Price Ratio   : {dispersion['q75_price_ratio']:.3f}x")
    print(f"  Interquartile Range (IQR)          : {dispersion['iqr_price_ratio']:.3f}")
    print(f"  Groups with Price Spread > 25%     : {dispersion['groups_with_price_spread_gt_25_pct_count']:,} "
          f"({dispersion['groups_with_price_spread_gt_25_pct_rate']:.1f}% of market)")
    print("-" * 78 + "\n")


if __name__ == "__main__":
    run_benchmark()
