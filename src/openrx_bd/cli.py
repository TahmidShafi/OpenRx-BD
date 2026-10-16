"""Command Line Interface for OpenRx-BD: Prescription Cost Optimizer."""

import argparse
import sys
from openrx_bd.engine import CostOptEngine
from openrx_bd.ingest import verify_schema
from openrx_bd.split import create_deterministic_splits


def format_currency(val: float | None) -> str:
    """Safely format currency with fallback if terminal doesn't support unicode."""
    if val is None:
        return "N/A"
    try:
        return f"৳ {val:.2f}"
    except UnicodeEncodeError:
        return f"BDT {val:.2f}"


def handle_optimize(args: argparse.Namespace) -> None:
    """Execute prescription cost optimization for a given brand."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print(f"\n=======================================================")
    print(f"        OpenRx-BD: Prescription Cost Optimizer         ")
    print(f"=======================================================\n")
    print(f"[Query] Searching for: '{args.brand}' (Dosage: {args.dosage or 'Any'}, Strength: {args.strength or 'Any'})\n")

    engine = CostOptEngine()
    res = engine.optimize_prescription(
        brand_query=args.brand,
        dosage_form=args.dosage,
        strength=args.strength,
        top_n=args.top_n,
    )

    if not res.query_generic:
        print(f"[-] {res.status_note}\n")
        return

    print("--- PRESCRIPTION DETAILS ---")
    print(f"  Brand Name     : {res.query_brand_name}")
    print(f"  Generic        : {res.query_generic}")
    print(f"  Dosage Form    : {res.query_dosage_form}")
    print(f"  Strength       : {res.query_strength or 'N/A'}")
    print(f"  Manufacturer   : {res.query_manufacturer}")
    print(f"  Current Price  : {format_currency(res.query_unit_price_bdt)} / unit")
    print(f"  Status         : {res.status_note}\n")

    if not res.alternatives:
        if res.is_already_optimal:
            print("[✓] Optimal Choice: No cheaper bioequivalent alternatives exist.")
        else:
            print("[i] No cheaper alternative pricing records available.")
        print()
        return

    print(f"--- TOP {len(res.alternatives)} BIOEQUIVALENT COST-SAVING ALTERNATIVES ---")
    header = f"{'Rank':<5} | {'Brand Name':<20} | {'Manufacturer':<25} | {'Unit Price':<12} | {'Savings':<10}"
    print(header)
    print("-" * len(header))

    for idx, alt in enumerate(res.alternatives, 1):
        price_str = format_currency(alt.unit_price_bdt)
        savings_str = f"-{alt.savings_pct:.1f}%"
        print(f"{idx:<5} | {alt.brand_name:<20} | {alt.manufacturer:<25} | {price_str:<12} | {savings_str:<10}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="openrx-bd",
        description="Prescription cost-optimizer for Bangladeshi pharmaceuticals",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and verify dataset schema")
    ingest_parser.add_argument(
        "--verify-schema",
        action="store_true",
        default=True,
        help="Verify the exact 10-column schema",
    )

    # Split subcommand
    split_parser = subparsers.add_parser("split", help="Generate train and blind-test splits")
    split_parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.80,
        help="Ratio of development set (default: 0.80)",
    )

    # Optimize subcommand
    opt_parser = subparsers.add_parser("optimize", help="Find cheaper bioequivalent substitutes")
    opt_parser.add_argument("--brand", required=True, help="Brand name to optimize (e.g. Napa, Ace)")
    opt_parser.add_argument("--dosage", default=None, help="Dosage form (e.g. Tablet, Syrup)")
    opt_parser.add_argument("--strength", default=None, help="Strength (e.g. 500 mg)")
    opt_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Maximum number of alternatives to return (default: 5)",
    )

    args = parser.parse_args()

    if args.subcommand == "ingest":
        verify_schema()
    elif args.subcommand == "split":
        create_deterministic_splits(dev_ratio=args.dev_ratio)
    elif args.subcommand == "optimize":
        handle_optimize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
