"""Deterministic generic substitution and prescription cost-optimization engine."""

from typing import Any
import pandas as pd
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process
from openrx_bd.parser import ParsedSuccess, parse_package_container
from openrx_bd.split import load_dev_split


class SubstitutionAlternative(BaseModel):
    """A verified, bioequivalent, cheaper brand alternative."""

    brand_id: str
    brand_name: str
    manufacturer: str
    dosage_form: str
    strength: str | None
    unit_price_bdt: float
    pack_price_bdt: float | None
    pack_size: int
    savings_pct: float = Field(description="Percentage savings compared to query brand")
    savings_per_unit_bdt: float


class OptimizationResult(BaseModel):
    """Result of prescription cost-optimization query."""

    query_brand_id: str
    query_brand_name: str
    query_generic: str
    query_dosage_form: str
    query_strength: str | None
    query_manufacturer: str
    query_unit_price_bdt: float | None

    is_already_optimal: bool
    status_note: str
    alternatives: list[SubstitutionAlternative] = Field(default_factory=list)


def normalize_str(val: Any) -> str:
    """Normalize string by trimming whitespace and lowercasing."""
    if val is None or pd.isna(val):
        return ""
    return str(val).strip().lower()


class CostOptEngine:
    """Prescription cost optimizer built on verified dataset splits."""

    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = load_dev_split()
        self.df = df.copy()
        self._prepare_index()

    def _prepare_index(self) -> None:
        """Cache normalized columns and parse prices once for speed."""
        self.df["norm_brand"] = self.df["brand name"].apply(normalize_str)
        self.df["norm_generic"] = self.df["generic"].apply(normalize_str)
        self.df["norm_dosage"] = self.df["dosage form"].apply(normalize_str)
        self.df["norm_strength"] = self.df["strength"].apply(normalize_str)
        self.df["norm_type"] = self.df["type"].apply(normalize_str)

        # Parse prices using strict parser
        parsed_results = [parse_package_container(x) for x in self.df["package container"]]
        self.df["parsed_unit_price"] = [
            r.unit_price_bdt if isinstance(r, ParsedSuccess) else None for r in parsed_results
        ]
        self.df["parsed_pack_price"] = [
            r.pack_price_bdt if isinstance(r, ParsedSuccess) else None for r in parsed_results
        ]
        self.df["parsed_pack_size"] = [
            r.pack_size if isinstance(r, ParsedSuccess) else 1 for r in parsed_results
        ]

        self.brand_lookup = dict(zip(self.df["norm_brand"], self.df["brand id"].astype(str)))

    def find_brand_records(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Fuzzy match a brand query and return candidate rows."""
        q_norm = normalize_str(query)
        # Direct exact match first
        exact_matches = self.df[self.df["norm_brand"] == q_norm]
        if not exact_matches.empty:
            return exact_matches.to_dict(orient="records")

        # Fuzzy match over unique brands
        unique_brands = self.df["brand name"].unique().tolist()
        fuzzy_matches = process.extract(
            query,
            unique_brands,
            scorer=fuzz.token_set_ratio,
            limit=limit,
        )
        matched_names = [m[0] for m in fuzzy_matches if m[1] >= 65]
        if not matched_names:
            return []

        matched_df = self.df[self.df["brand name"].isin(matched_names)]
        return matched_df.to_dict(orient="records")

    def optimize_prescription(
        self,
        brand_query: str,
        dosage_form: str | None = None,
        strength: str | None = None,
        top_n: int = 5,
    ) -> OptimizationResult:
        """Find cheaper bioequivalent substitutes with identical generic, dosage form, and strength."""
        candidates = self.find_brand_records(brand_query)
        if not candidates:
            return OptimizationResult(
                query_brand_id="",
                query_brand_name=brand_query,
                query_generic="",
                query_dosage_form="",
                query_strength=None,
                query_manufacturer="",
                query_unit_price_bdt=None,
                is_already_optimal=False,
                status_note=f"No matching brand found for '{brand_query}' in dataset.",
            )

        # Filter by dosage form and strength if provided
        matched_row = candidates[0]
        if dosage_form or strength:
            for row in candidates:
                dosage_match = (
                    not dosage_form or normalize_str(dosage_form) in normalize_str(row["dosage form"])
                )
                strength_match = (
                    not strength or normalize_str(strength) == normalize_str(row["strength"])
                )
                if dosage_match and strength_match:
                    matched_row = row
                    break

        query_generic = matched_row["generic"]
        query_dosage = matched_row["dosage form"]
        query_strength = matched_row["strength"]
        query_type = matched_row["type"]
        query_brand_id = str(matched_row["brand id"])
        query_brand_name = matched_row["brand name"]
        query_mfr = matched_row["manufacturer"]
        query_price = matched_row["parsed_unit_price"]

        # Bioequivalence filter: identical generic, dosage form, strength, and type
        pool_mask = (
            (self.df["norm_generic"] == normalize_str(query_generic))
            & (self.df["norm_dosage"] == normalize_str(query_dosage))
            & (self.df["norm_strength"] == normalize_str(query_strength))
            & (self.df["norm_type"] == normalize_str(query_type))
            & (self.df["parsed_unit_price"].notna())
        )
        pool = self.df[pool_mask].copy()

        if pool.empty:
            return OptimizationResult(
                query_brand_id=query_brand_id,
                query_brand_name=query_brand_name,
                query_generic=query_generic,
                query_dosage_form=query_dosage,
                query_strength=query_strength,
                query_manufacturer=query_mfr,
                query_unit_price_bdt=query_price,
                is_already_optimal=True,
                status_note="No alternative brands with verified pricing found in this bioequivalent class.",
            )

        # Sort pool by unit price ascending
        pool = pool.sort_values(by="parsed_unit_price", ascending=True)

        if query_price is None or pd.isna(query_price):
            status_note = "Query brand has unlisted price; showing lowest available generic alternatives."
            query_price_val = None
        else:
            query_price_val = float(query_price)
            status_note = "Optimization successful."

        alternatives: list[SubstitutionAlternative] = []
        for _, alt in pool.iterrows():
            alt_id = str(alt["brand id"])
            if alt_id == query_brand_id:
                continue

            alt_price = float(alt["parsed_unit_price"])
            if query_price_val is not None:
                savings_per_unit = max(0.0, query_price_val - alt_price)
                savings_pct = (
                    (savings_per_unit / query_price_val * 100.0) if query_price_val > 0 else 0.0
                )
            else:
                savings_per_unit = 0.0
                savings_pct = 0.0

            # Only suggest cheaper or equivalent items if price is known
            if query_price_val is not None and alt_price >= query_price_val:
                continue

            alternatives.append(
                SubstitutionAlternative(
                    brand_id=alt_id,
                    brand_name=alt["brand name"],
                    manufacturer=alt["manufacturer"],
                    dosage_form=alt["dosage form"],
                    strength=alt["strength"],
                    unit_price_bdt=alt_price,
                    pack_price_bdt=alt["parsed_pack_price"],
                    pack_size=int(alt["parsed_pack_size"]),
                    savings_pct=round(savings_pct, 1),
                    savings_per_unit_bdt=round(savings_per_unit, 2),
                )
            )
            if len(alternatives) >= top_n:
                break

        is_optimal = len(alternatives) == 0 and query_price_val is not None
        if is_optimal:
            status_note = f"'{query_brand_name}' is already the most cost-effective brand in this bioequivalent group."

        return OptimizationResult(
            query_brand_id=query_brand_id,
            query_brand_name=query_brand_name,
            query_generic=query_generic,
            query_dosage_form=query_dosage,
            query_strength=query_strength,
            query_manufacturer=query_mfr,
            query_unit_price_bdt=query_price_val,
            is_already_optimal=is_optimal,
            status_note=status_note,
            alternatives=alternatives,
        )
