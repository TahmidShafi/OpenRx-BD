"""Container parser implementing strict grammar extraction, invariant checks, and cross-column dimensional reconciliation."""

import re
from typing import Any
import polars as pl
from src.parser.schemas import DispensingCategory, PackSKU, RowParseResult

CONTAINER_TYPES = [
    "bottle",
    "vial",
    "ampoule",
    "tube",
    "drop",
    "sachet",
    "syringe",
    "inhaler",
    "cartridge",
    "pot",
    "can",
    "tin",
    "jar",
    "bag",
    "box",
    "strip",
    "container",
    "device",
    "flexpen",
    "pen",
    "unit",
]

MEASUREMENT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ml|mg|gm|g|mcg|iu|billion|metered doses|doses|metered sprays|sprays|puffs)\b",
    re.IGNORECASE,
)

# Regex to split multi-SKU strings on comma when followed by another container spec, NOT pack info
CONTAINER_SPLIT_PATTERN = re.compile(
    r",(?!\s*\(\s*\d+)(?=\s*(?:(?:\d+(?:\.\d+)?\s*(?:ml|mg|gm|g|mcg|iu|billion|vial|bottle|tube|ampoule|drop|container|bag|jar|pack|pot|can|tin|solution|suspension|spray|metered))\b|[a-zA-Z][^:,\n]*:\s*৳))",
    re.IGNORECASE,
)

# Regex 1: Composite unit + pack
# Example: "Unit Price: ৳ 6.50,(15's pack: ৳ 97.50)"
# Example: "500 mg vial: ৳ 28.43,(5's pack: ৳ 142.15)"
PATTERN_COMPOSITE = re.compile(
    r"(?:(?P<container_spec>[^:৳,\n]+):\s*)?৳\s*(?P<unit_price>[\d,]+(?:\.\d+)?)\s*,\s*\(\s*(?P<pack_qty>\d+)\s*(?:'s)?\s*pack:\s*৳\s*(?P<pack_price>[\d,]+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)

# Regex 2: Single-tier container price
# Example: "100 ml bottle: ৳ 60.00" or "450 ml bottle: ৳ 200.00"
PATTERN_SINGLE_CONTAINER = re.compile(
    r"(?P<container_spec>[^:৳,\n]+):\s*৳\s*(?P<price>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Regex 3: Standalone unit price
# Example: "Unit Price: ৳ 200.00" or "৳ 12.50"
PATTERN_STANDALONE_UNIT = re.compile(
    r"(?:Unit Price:\s*)?৳\s*(?P<price>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Regex 4: Standalone pack price
# Example: "(30's pack: ৳ 150.00)"
PATTERN_STANDALONE_PACK = re.compile(
    r"\(\s*(?P<pack_qty>\d+)\s*(?:'s)?\s*pack:\s*৳\s*(?P<pack_price>[\d,]+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)


def classify_dispensing_category(dosage_form: str | None) -> DispensingCategory:
    """Classify dosage form into DIVISIBLE_SOLID, INDIVISIBLE_CONTAINER, or AMBIGUOUS_FORM."""
    if not dosage_form:
        return DispensingCategory.AMBIGUOUS_FORM
    f = str(dosage_form).strip().lower()

    # Indivisible liquid / container keyword indicators
    indivisible_keywords = [
        "syrup",
        "suspension",
        "solution",
        "bottle",
        "vial",
        "ampoule",
        "injection",
        "infusion",
        "drop",
        "spray",
        "inhaler",
        "cream",
        "ointment",
        "gel",
        "lotion",
        "shampoo",
        "liniment",
        "emulsion",
        "liquid",
        "enema",
        "mouthwash",
        "gargle",
        "paint",
        "paste",
        "soap",
        "wash",
        "hand rub",
        "powder for suspension",
        "powder for solution",
        "transdermal",
    ]
    if any(k in f for k in indivisible_keywords):
        return DispensingCategory.INDIVISIBLE_CONTAINER

    # Divisible solid indicators
    divisible_keywords = [
        "tablet",
        "capsule",
        "caplet",
        "pellet",
        "lozenge",
        "suppository",
        "bolus",
        "pill",
        "sachet",
    ]
    if any(k in f for k in divisible_keywords):
        return DispensingCategory.DIVISIBLE_SOLID

    return DispensingCategory.AMBIGUOUS_FORM


def extract_container_and_measurement(spec: str | None) -> tuple[str, float | None, str | None]:
    """Extract container type, measurement numeric value, and unit from a container spec string."""
    if not spec:
        return "unit", None, None
    s = str(spec).strip()
    if s.lower() == "unit price":
        return "unit", None, None

    m_val: float | None = None
    m_unit: str | None = None
    m = MEASUREMENT_PATTERN.search(s)
    if m:
        m_val = float(m.group(1))
        m_unit = m.group(2).lower()

    ctype: str | None = None
    s_lower = s.lower()
    for ct in CONTAINER_TYPES:
        if ct in s_lower:
            ctype = ct
            break
    if not ctype:
        ctype = s_lower if s else "unit"

    return ctype, m_val, m_unit


def parse_concentration_mg_per_ml(strength: str | None) -> float | None:
    """Extract drug concentration in mg/ml from a strength string."""
    if not strength:
        return None
    s = str(strength).strip()

    # Pattern: X mg/ml or X mg/Y ml
    m = re.search(r"(\d+(?:\.\d+)?)\s*mg\s*/\s*(\d+(?:\.\d+)?)?\s*ml", s, re.IGNORECASE)
    if m:
        mass = float(m.group(1))
        vol = float(m.group(2)) if m.group(2) else 1.0
        return mass / vol if vol > 0 else None

    # Pattern: (X mg+Y mg)/Z ml
    m2 = re.search(r"\(([^)]+)\)\s*/\s*(\d+(?:\.\d+)?)?\s*ml", s, re.IGNORECASE)
    if m2:
        inner = m2.group(1)
        vol = float(m2.group(2)) if m2.group(2) else 1.0
        mg_vals = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*mg", inner, re.IGNORECASE)]
        if mg_vals and vol > 0:
            return sum(mg_vals) / vol

    return None


def reconcile_dimensions(
    strength: str | None,
    skus: list[PackSKU],
) -> float | None:
    """Implement cross-column dimensional reconciliation.

    When strength is a concentration (e.g. '25 mg/ml') and package container states
    mass (e.g. '250 mg vial'), compute volume = 250 / 25 = 10.0 ml.
    """
    if not skus:
        return None

    conc = parse_concentration_mg_per_ml(strength)
    primary_sku = skus[0]

    # Case 1: Concentration in strength, mass in package container
    if conc is not None and conc > 0 and primary_sku.measurement_value is not None:
        unit = (primary_sku.measurement_unit or "").lower()
        if unit == "mg":
            return round(primary_sku.measurement_value / conc, 4)
        if unit in ["gm", "g"]:
            return round((primary_sku.measurement_value * 1000.0) / conc, 4)

    # Case 2: Direct volume in container specification (e.g., 100 ml bottle)
    if primary_sku.measurement_unit == "ml" and primary_sku.measurement_value is not None:
        return round(primary_sku.measurement_value, 4)

    return None


def parse_sku_chunk(chunk: str, sku_index: int) -> tuple[PackSKU | None, bool]:
    """Parse a single SKU string chunk into a PackSKU model.

    Returns (PackSKU, invariant_failure_flag).
    """
    text = chunk.strip().rstrip(",")
    if not text:
        return None, False

    # 1. Composite unit + pack
    m_comp = PATTERN_COMPOSITE.search(text)
    if m_comp:
        u = float(m_comp.group("unit_price").replace(",", ""))
        q = int(m_comp.group("pack_qty"))
        p = float(m_comp.group("pack_price").replace(",", ""))
        spec = m_comp.group("container_spec")
        ctype, m_val, m_unit = extract_container_and_measurement(spec)

        diff = abs(u * q - p)
        passed = diff <= 0.05
        sku = PackSKU(
            sku_index=sku_index,
            container_type=ctype,
            measurement_value=m_val,
            measurement_unit=m_unit,
            unit_price_bdt=u,
            pack_qty=q,
            pack_price_bdt=p,
            effective_per_unit_bdt=u,
            invariant_checked=True,
            invariant_passed=passed,
        )
        return sku, not passed

    # 2. Single container price (e.g. 100 ml bottle: ৳ 60.00)
    m_single = PATTERN_SINGLE_CONTAINER.search(text)
    if m_single:
        p = float(m_single.group("price").replace(",", ""))
        spec = m_single.group("container_spec")
        ctype, m_val, m_unit = extract_container_and_measurement(spec)

        sku = PackSKU(
            sku_index=sku_index,
            container_type=ctype,
            measurement_value=m_val,
            measurement_unit=m_unit,
            unit_price_bdt=p,
            pack_qty=None,
            pack_price_bdt=p,
            effective_per_unit_bdt=p,
            invariant_checked=False,
            invariant_passed=None,
        )
        return sku, False

    # 3. Standalone pack price (e.g. (30's pack: ৳ 150.00))
    m_pack = PATTERN_STANDALONE_PACK.search(text)
    if m_pack:
        q = int(m_pack.group("pack_qty"))
        p = float(m_pack.group("pack_price").replace(",", ""))
        eff = round(p / q, 4) if q > 0 else p
        sku = PackSKU(
            sku_index=sku_index,
            container_type="unit",
            measurement_value=None,
            measurement_unit=None,
            unit_price_bdt=eff,
            pack_qty=q,
            pack_price_bdt=p,
            effective_per_unit_bdt=eff,
            invariant_checked=False,
            invariant_passed=None,
        )
        return sku, False

    # 4. Standalone unit price (e.g. Unit Price: ৳ 200.00)
    m_unit = PATTERN_STANDALONE_UNIT.search(text)
    if m_unit:
        p = float(m_unit.group("price").replace(",", ""))
        sku = PackSKU(
            sku_index=sku_index,
            container_type="unit",
            measurement_value=None,
            measurement_unit=None,
            unit_price_bdt=p,
            pack_qty=None,
            pack_price_bdt=None,
            effective_per_unit_bdt=p,
            invariant_checked=False,
            invariant_passed=None,
        )
        return sku, False

    return None, False


def parse_container(
    brand_id: str | int,
    package_container: str | None,
    dosage_form: str | None = None,
    strength: str | None = None,
) -> RowParseResult:
    """Parse package container text into a validated RowParseResult."""
    bid = str(brand_id)
    dispensing_cat = classify_dispensing_category(dosage_form)

    # Check null / empty
    if package_container is None or not str(package_container).strip():
        return RowParseResult(
            brand_id=bid,
            status="ABSTAIN_NULL_CONTAINER",
            dispensing_category=dispensing_cat,
            abstain_reason="Package container is null or empty",
            skus=[],
        )

    raw = str(package_container).strip()
    raw_lower = raw.lower()
    if "unavailable" in raw_lower or "not for sale" in raw_lower:
        return RowParseResult(
            brand_id=bid,
            status="ABSTAIN_UNSUPPORTED_TEMPLATE",
            dispensing_category=dispensing_cat,
            abstain_reason="Pricing marked unavailable or not for sale",
            skus=[],
        )

    if "৳" not in raw:
        return RowParseResult(
            brand_id=bid,
            status="ABSTAIN_UNSUPPORTED_TEMPLATE",
            dispensing_category=dispensing_cat,
            abstain_reason="Missing currency symbol ৳",
            skus=[],
        )

    # 1. Split on double commas ,, first
    double_comma_chunks = [c.strip() for c in raw.split(",,") if c.strip()]
    sku_chunks: list[str] = []
    for dc in double_comma_chunks:
        # 2. Split on single comma preceding another container
        sub_chunks = [
            sc.strip().rstrip(",")
            for sc in CONTAINER_SPLIT_PATTERN.split(dc)
            if sc.strip().rstrip(",")
        ]
        sku_chunks.extend(sub_chunks)

    if not sku_chunks:
        return RowParseResult(
            brand_id=bid,
            status="ABSTAIN_UNSUPPORTED_TEMPLATE",
            dispensing_category=dispensing_cat,
            abstain_reason="No parseable SKU segments found",
            skus=[],
        )

    parsed_skus: list[PackSKU] = []
    has_invariant_violation = False
    violation_detail = ""

    for idx, chunk in enumerate(sku_chunks):
        sku, inv_failed = parse_sku_chunk(chunk, sku_index=idx)
        if sku is None:
            # Unparseable chunk
            continue
        parsed_skus.append(sku)
        if inv_failed:
            has_invariant_violation = True
            diff = abs(
                (sku.unit_price_bdt or 0) * (sku.pack_qty or 1) - (sku.pack_price_bdt or 0)
            )
            violation_detail = (
                f"Multiplicative invariant failed in SKU {sku.sku_index}: "
                f"|{sku.unit_price_bdt} * {sku.pack_qty} - {sku.pack_price_bdt}| = {diff:.4f} > 0.05"
            )

    if not parsed_skus:
        return RowParseResult(
            brand_id=bid,
            status="ABSTAIN_UNSUPPORTED_TEMPLATE",
            dispensing_category=dispensing_cat,
            abstain_reason="No recognized grammar pattern matched in string",
            skus=[],
        )

    if has_invariant_violation:
        return RowParseResult(
            brand_id=bid,
            status="INVARIANT_VIOLATION",
            dispensing_category=dispensing_cat,
            abstain_reason=violation_detail,
            skus=parsed_skus,
        )

    # Cross-column dimensional reconciliation
    reconciled_vol = reconcile_dimensions(strength, parsed_skus)

    return RowParseResult(
        brand_id=bid,
        status="PARSED_VALID",
        dispensing_category=dispensing_cat,
        reconciled_total_volume_ml=reconciled_vol,
        skus=parsed_skus,
    )


def parse_row(row: dict[str, Any]) -> RowParseResult:
    """Parse a single dictionary row containing brand id, package container, dosage form, strength."""
    return parse_container(
        brand_id=row.get("brand id", ""),
        package_container=row.get("package container"),
        dosage_form=row.get("dosage form"),
        strength=row.get("strength"),
    )


def parse_dataframe(df: pl.DataFrame) -> list[RowParseResult]:
    """Parse all rows of a Polars DataFrame into RowParseResults."""
    results: list[RowParseResult] = []
    for row in df.iter_rows(named=True):
        results.append(parse_row(row))
    return results


def audit_dev_split(dev_split_path: str = "data/splits/dev_split.parquet") -> dict[str, int]:
    """Audit parser status distribution over dev_split.parquet."""
    df = pl.read_parquet(dev_split_path)
    results = parse_dataframe(df)

    from collections import Counter
    status_counts = Counter(r.status for r in results)
    total = len(results)

    print("\n" + "=" * 62)
    print("Package Container Parser Audit over data/splits/dev_split.parquet")
    print("=" * 62)
    print(f"Total Rows Audited: {total:,}\n")

    for status_key in [
        "PARSED_VALID",
        "INVARIANT_VIOLATION",
        "ABSTAIN_UNSUPPORTED_TEMPLATE",
        "ABSTAIN_NULL_CONTAINER",
    ]:
        count = status_counts.get(status_key, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  - {status_key:<28} : {count:6,} ({pct:5.2f}%)")

    print("=" * 62 + "\n")
    return dict(status_counts)


if __name__ == "__main__":
    audit_dev_split()
