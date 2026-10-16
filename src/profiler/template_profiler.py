"""Structural template profiler for Bangladeshi pharmaceutical package containers."""

from collections import Counter
import json
from pathlib import Path
import re
import sys
import polars as pl

PARQUET_PATH = Path("data/raw/medicines.parquet")
REPORTS_DIR = Path("reports")
REPORT_JSON_PATH = REPORTS_DIR / "template_profile.json"


def normalize_sku_chunk(chunk: str) -> str:
    """Normalize a single package container SKU segment into a structural template.

    Rules applied:
    b) Replace all numbers after ৳ with <PRICE>.
    c) Replace pack quantities like (15's pack with (<QTY>'s pack.
    d) Replace measurements (e.g., 100 ml, 500 mg, 2.5 ml) with <MEASUREMENT>.
    """
    s = chunk.strip()
    if not s:
        return ""

    # b) Replace all floating-point/integer numbers after ৳ with <PRICE>
    s = re.sub(r"৳\s*[\d,]+(?:\.\d+)?", "৳ <PRICE>", s)

    # c) Replace pack quantities like (15's pack with (<QTY>'s pack
    s = re.sub(r"\(\s*\d+\'?s?\s*pack", "(<QTY>'s pack", s, flags=re.IGNORECASE)

    # d) Replace measurements (leading or after commas, e.g. 100 ml, 500 mg, 2.5 ml, 100 gm) with <MEASUREMENT>
    s = re.sub(
        r"(?:^|(?<=,)|(?<=,\s))\s*\d+(?:\.\d+)?\s*(?:ml|mg|gm|g|mcg|iu|billion|metered sprays|puffs|doses)\b",
        "<MEASUREMENT>",
        s,
        flags=re.IGNORECASE,
    )

    return s


def profile_templates(parquet_path: Path = PARQUET_PATH) -> dict:
    """Read medicines.parquet, profile structural templates, and output report."""
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing dataset at {parquet_path}. Run fetch_dataset first.")

    df = pl.read_parquet(parquet_path)
    total_rows = len(df)
    null_rows = int(df["package container"].is_null().sum())
    non_null_rows = total_rows - null_rows

    row_templates: list[str] = []
    sku_templates: list[str] = []

    for raw in df["package container"].to_list():
        if raw is None or not str(raw).strip():
            row_templates.append("<NULL_OR_EMPTY>")
            continue

        # a) Split multi-SKU records on ,, first
        chunks = [c.strip() for c in str(raw).split(",,") if c.strip()]
        norm_chunks = [normalize_sku_chunk(c) for c in chunks]

        for nc in norm_chunks:
            sku_templates.append(nc)

        row_templates.append(",,".join(norm_chunks))

    # Calculate distributions
    row_counts = Counter(row_templates)
    # Remove null count from distinct pricing templates
    pricing_templates = {k: v for k, v in row_counts.items() if k != "<NULL_OR_EMPTY>"}
    distinct_templates_count = len(pricing_templates)

    # Top 15 templates
    top_15 = []
    cumulative_pct = 0.0
    for rank, (template, count) in enumerate(row_counts.most_common(15), start=1):
        pct = (count / total_rows) * 100.0
        cumulative_pct += pct
        top_15.append(
            {
                "rank": rank,
                "template": template,
                "count": count,
                "pct_share": round(pct, 2),
                "cumulative_pct": round(cumulative_pct, 2),
            }
        )

    # Prepare report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "dataset_path": str(parquet_path).replace("\\", "/"),
        "total_rows": total_rows,
        "non_null_rows": non_null_rows,
        "null_rows": null_rows,
        "distinct_templates_count": distinct_templates_count,
        "top_15_templates": top_15,
        "all_templates_distribution": dict(row_counts.most_common()),
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return report_data


def print_markdown_report(report_data: dict) -> None:
    """Print the formatted Markdown table and summary to stdout."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("\n## Package Container Structural Template Profile")
    print(f"- **Total Rows:** {report_data['total_rows']:,}")
    print(f"- **Non-Null Rows:** {report_data['non_null_rows']:,}")
    print(f"- **Null Rows:** {report_data['null_rows']:,}")
    print(f"- **Distinct Structural Templates Discovered:** {report_data['distinct_templates_count']:,}\n")

    print("| Rank | Structural Template | Row Count | % Share | Cumulative % |")
    print("| :--- | :------------------ | --------: | ------: | -----------: |")

    for item in report_data["top_15_templates"]:
        # Escape pipes if any exist in template for clean markdown rendering
        tmpl = item["template"].replace("|", "\\|")
        print(
            f"| {item['rank']} | `{tmpl}` | {item['count']:,} | {item['pct_share']:.2f}% | {item['cumulative_pct']:.2f}% |"
        )
    print()


def main() -> None:
    report_data = profile_templates()
    print_markdown_report(report_data)


if __name__ == "__main__":
    main()
