"""Debate-hardened, non-silent regex parser for Bangladeshi pharmaceutical package containers."""

from typing import Annotated, Literal, Union
import re
from pydantic import BaseModel, Field


class ParsedSuccess(BaseModel):
    """Successfully parsed packaging and unit-pricing."""

    status: Literal["success"] = "success"
    raw_text: str
    unit_price_bdt: float
    pack_price_bdt: float | None = None
    pack_size: int = 1
    container_type: str = "unit"


class ParsedAbstain(BaseModel):
    """Explicit abstention when data is absent, unavailable, or unsupported."""

    status: Literal["abstain"] = "abstain"
    raw_text: str
    reason: str


class InvariantViolation(BaseModel):
    """Detected mathematical or physical contradiction in the data."""

    status: Literal["invariant_violation"] = "invariant_violation"
    raw_text: str
    violation: str


ParseResult = Annotated[
    Union[ParsedSuccess, ParsedAbstain, InvariantViolation],
    Field(discriminator="status"),
]


# Regex 1: Container with unit price and pack info
# Example: "500 mg vial: ৳ 28.43,(5's pack: ৳ 142.15),"
# Example: "Unit Price: ৳ 5.98,(100's pack: ৳ 598.00),"
PATTERN_UNIT_AND_PACK = re.compile(
    r"(?:(?P<container>[^:৳,\n]+):\s*)?৳\s*(?P<unit_price>[\d,]+(?:\.\d+)?)"
    r"\s*,\s*\(\s*(?P<pack_size>\d+)'s\s*pack:\s*৳\s*(?P<pack_price>[\d,]+(?:\.\d+)?)\s*\)",
    re.IGNORECASE,
)

# Regex 2: Container with single unit price
# Example: "100 ml bottle: ৳ 70.00" or "Unit Price: ৳ 200.00"
PATTERN_CONTAINER_ONLY = re.compile(
    r"(?P<container>[^:৳,\n]+):\s*৳\s*(?P<unit_price>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Regex 3: Fallback standalone Taka price
PATTERN_STANDALONE_TAKA = re.compile(
    r"৳\s*(?P<unit_price>[\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def clean_number(val: str) -> float:
    """Safely convert numeric string with potential commas to float."""
    return float(val.replace(",", "").strip())


def parse_package_container(raw: str | None) -> ParseResult:
    """Parse raw package container string into a strict Pydantic union.

    NEVER raises unhandled exceptions. Returns ParsedSuccess, ParsedAbstain,
    or InvariantViolation.
    """
    if raw is None:
        return ParsedAbstain(raw_text="", reason="Package container is None/null")

    text = str(raw).strip()
    if not text:
        return ParsedAbstain(raw_text="", reason="Package container is empty")

    lower = text.lower()
    if "unavailable" in lower or "not for sale" in lower:
        return ParsedAbstain(raw_text=text, reason="Marked unavailable or not for sale")

    if "৳" not in text:
        return ParsedAbstain(raw_text=text, reason="No BDT (৳) currency symbol found")

    # If multiple entries exist separated by ,, or multiple container specs, take the primary (first)
    primary_chunk = text.split(",,")[0].strip()

    # 1. Try unit and pack combination pattern
    m_pack = PATTERN_UNIT_AND_PACK.search(primary_chunk)
    if m_pack:
        unit_p = clean_number(m_pack.group("unit_price"))
        pack_sz = int(m_pack.group("pack_size"))
        pack_p = clean_number(m_pack.group("pack_price"))
        container = (m_pack.group("container") or "unit").strip()
        if container.lower() == "unit price":
            container = "unit"

        # Check invariants
        if unit_p <= 0:
            return InvariantViolation(
                raw_text=text, violation=f"Unit price must be positive, got {unit_p}"
            )
        if pack_sz < 1:
            return InvariantViolation(
                raw_text=text, violation=f"Pack size must be >= 1, got {pack_sz}"
            )
        if pack_p <= 0:
            return InvariantViolation(
                raw_text=text, violation=f"Pack price must be positive, got {pack_p}"
            )

        # Mathematical consistency check (allowing small rounding difference)
        expected_pack = unit_p * pack_sz
        discrepancy = abs(expected_pack - pack_p)
        max_allowed_diff = max(2.5, 0.08 * pack_p)
        if discrepancy > max_allowed_diff:
            return InvariantViolation(
                raw_text=text,
                violation=(
                    f"Price mismatch: {unit_p} * {pack_sz} = {expected_pack:.2f} BDT "
                    f"differs from pack price {pack_p:.2f} BDT by {discrepancy:.2f} BDT"
                ),
            )

        return ParsedSuccess(
            raw_text=text,
            unit_price_bdt=round(unit_p, 4),
            pack_price_bdt=round(pack_p, 2),
            pack_size=pack_sz,
            container_type=container,
        )

    # 2. Try single container price pattern
    m_single = PATTERN_CONTAINER_ONLY.search(primary_chunk)
    if m_single:
        unit_p = clean_number(m_single.group("unit_price"))
        container = m_single.group("container").strip()
        if container.lower() == "unit price":
            container = "unit"

        if unit_p <= 0:
            return InvariantViolation(
                raw_text=text, violation=f"Unit price must be positive, got {unit_p}"
            )

        return ParsedSuccess(
            raw_text=text,
            unit_price_bdt=round(unit_p, 4),
            pack_price_bdt=round(unit_p, 4),
            pack_size=1,
            container_type=container,
        )

    # 3. Fallback standalone Taka
    m_taka = PATTERN_STANDALONE_TAKA.search(primary_chunk)
    if m_taka:
        unit_p = clean_number(m_taka.group("unit_price"))
        if unit_p <= 0:
            return InvariantViolation(
                raw_text=text, violation=f"Unit price must be positive, got {unit_p}"
            )
        return ParsedSuccess(
            raw_text=text,
            unit_price_bdt=round(unit_p, 4),
            pack_price_bdt=round(unit_p, 4),
            pack_size=1,
            container_type="unit",
        )

    return ParsedAbstain(raw_text=text, reason="Unrecognized pricing format")
