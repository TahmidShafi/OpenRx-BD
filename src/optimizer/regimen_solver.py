"""Pharmaceutical regimen optimization engine with polypharmacy safety audit and retail dispensing knapsack solver."""

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Literal, Optional
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from src.ontology.entity_resolver import (
    PharmaceuticalEquivalenceKey,
    ResolvedEntity,
    SafetyAlert,
    check_prescription_safety,
    resolve_medicine_entity,
)
from src.parser.container_parser import parse_container, parse_dataframe
from src.parser.schemas import DispensingCategory, PackSKU, RowParseResult


class PrescriptionItem(BaseModel):
    """Specification of an individual medication item within a clinical prescription."""

    model_config = ConfigDict(extra="forbid")

    brand_id: str = Field(..., description="Unique brand identifier or exact brand name")
    units_per_dose: float = Field(..., gt=0, description="Dose volume or count per administration (e.g. 1 tab, 10 ml)")
    doses_per_day: float = Field(..., gt=0, description="Frequency of doses per 24-hour period (e.g. 2)")
    duration_days: int = Field(..., gt=0, description="Prescription duration in calendar days (e.g. 30)")

    @property
    def total_units_needed(self) -> float:
        """Total quantity of units required across the full treatment course."""
        return round(self.units_per_dose * self.doses_per_day * self.duration_days, 4)


class ContainerChoice(BaseModel):
    """Breakdown of an individual container or package unit selected for dispensing."""

    model_config = ConfigDict(extra="forbid")

    container_name: str
    capacity: float
    unit: str
    unit_price_bdt: float
    quantity: int
    subtotal_bdt: float


class DispensingPlan(BaseModel):
    """Dispensing configuration, total cost, and waste for a specific brand."""

    model_config = ConfigDict(extra="forbid")

    strategy: Literal["CUT_STRIP", "FULL_PACK", "INDIVISIBLE_CONTAINER", "UNIT_FALLBACK"]
    target_units_needed: float
    total_units_supplied: float
    leftover_waste: float
    total_cost_bdt: float
    unit_of_measure: str
    container_breakdown: list[ContainerChoice] = Field(default_factory=list)
    details: str = ""


class BrandCandidate(BaseModel):
    """Evaluated bioequivalent brand candidate with dispensing economics."""

    model_config = ConfigDict(extra="forbid")

    brand_id: str
    brand_name: str
    manufacturer: str
    dosage_form: str
    dispensing_category: DispensingCategory
    total_cost_bdt: float
    leftover_waste: float
    dispensing_plan: DispensingPlan


class ItemOptimizationResult(BaseModel):
    """Optimization and safety evaluation result for a single prescribed drug item."""

    model_config = ConfigDict(extra="forbid")

    prescribed_item: PrescriptionItem
    brand_id: str
    brand_name: str
    generic_name: str
    dosage_form: str
    strength: Optional[str] = None
    dispensing_category: DispensingCategory
    total_units_needed: float
    is_nti_locked: bool
    is_substitutable: bool
    substitution_block_reason: Optional[str] = None

    prescribed_cost_bdt: float
    prescribed_waste: float
    prescribed_plan: DispensingPlan

    optimal_brand_id: str
    optimal_brand_name: str
    optimal_manufacturer: str
    optimal_cost_bdt: float
    optimal_waste: float
    optimal_plan: DispensingPlan

    savings_bdt: float
    savings_pct: float
    num_candidates: int
    candidates: list[BrandCandidate] = Field(default_factory=list)


class RegimenOptimizationResult(BaseModel):
    """Comprehensive multi-drug prescription regimen optimization and safety audit."""

    model_config = ConfigDict(extra="forbid")

    prescribed_total_cost_bdt: float
    optimized_total_cost_bdt: float
    total_savings_bdt: float
    total_savings_pct: float
    safety_alerts: list[SafetyAlert] = Field(default_factory=list)
    has_duplicate_ingredients: bool = False
    has_nti_lock: bool = False
    items: list[ItemOptimizationResult] = Field(default_factory=list)


# ==============================================================================
# 1. Unbounded Knapsack DP for Indivisible Containers (Syrups, Inhalers, Drops)
# ==============================================================================

def get_scale_factor(numbers: list[float]) -> int:
    """Determine dynamic integer scaling factor to eliminate float precision loss in DP."""
    if all(abs(x - round(x)) < 1e-4 for x in numbers):
        return 1
    if all(abs(x * 10 - round(x * 10)) < 1e-3 for x in numbers):
        return 10
    return 100


def solve_indivisible_knapsack(
    target_units: float,
    containers: list[tuple[float, float, str]],  # (capacity, cost_bdt, name)
    unit_name: str = "ml",
) -> DispensingPlan:
    """Solve exact minimum-cost unbounded knapsack to cover target_units with indivisible containers.

    Picks combination of container sizes to cover >= target_units at minimum BDT cost.
    Ties are broken by minimizing leftover waste volume.
    """
    if target_units <= 0:
        return DispensingPlan(
            strategy="INDIVISIBLE_CONTAINER",
            target_units_needed=0.0,
            total_units_supplied=0.0,
            leftover_waste=0.0,
            total_cost_bdt=0.0,
            unit_of_measure=unit_name,
            details="Zero quantity required",
        )

    valid_containers = [(cap, cost, name) for cap, cost, name in containers if cap > 0 and cost > 0]
    if not valid_containers:
        # Fallback if no container capacity is known: treat 1 container as needed
        return DispensingPlan(
            strategy="INDIVISIBLE_CONTAINER",
            target_units_needed=target_units,
            total_units_supplied=target_units,
            leftover_waste=0.0,
            total_cost_bdt=0.0,
            unit_of_measure=unit_name,
            details="No container sizing available for exact knapsack calculation",
        )

    all_nums = [target_units] + [cap for cap, _, _ in valid_containers]
    scale = get_scale_factor(all_nums)

    target_int = int(round(target_units * scale))
    scaled_items = [
        (int(round(cap * scale)), cost, cap, name, idx)
        for idx, (cap, cost, name) in enumerate(valid_containers)
    ]

    max_cap_int = max(item[0] for item in scaled_items)
    limit = target_int + max_cap_int

    # dp[v] = minimum cost to achieve exact integer volume v
    dp = [float("inf")] * (limit + 1)
    dp[0] = 0.0
    parent: list[Optional[tuple[int, int]]] = [None] * (limit + 1)

    for v in range(limit + 1):
        if dp[v] == float("inf"):
            continue
        for cap_int, cost, _, _, idx in scaled_items:
            nv = v + cap_int
            if nv <= limit:
                nc = dp[v] + cost
                if nc < dp[nv] - 1e-6:
                    dp[nv] = nc
                    parent[nv] = (v, idx)

    # Find minimum cost covering at least target_int, with tie-break on minimal waste
    best_cost = float("inf")
    best_v: Optional[int] = None
    for v in range(target_int, limit + 1):
        cost = dp[v]
        if cost < best_cost - 1e-5:
            best_cost = cost
            best_v = v
        elif abs(cost - best_cost) <= 1e-5 and best_v is not None and v < best_v:
            best_v = v

    if best_v is None or best_cost == float("inf"):
        # Extreme fallback
        primary_cap, primary_cost, primary_name = valid_containers[0]
        qty = math.ceil(target_units / primary_cap)
        tot_units = qty * primary_cap
        cost = qty * primary_cost
        return DispensingPlan(
            strategy="INDIVISIBLE_CONTAINER",
            target_units_needed=target_units,
            total_units_supplied=tot_units,
            leftover_waste=round(tot_units - target_units, 4),
            total_cost_bdt=round(cost, 2),
            unit_of_measure=unit_name,
            container_breakdown=[
                ContainerChoice(
                    container_name=primary_name,
                    capacity=primary_cap,
                    unit=unit_name,
                    unit_price_bdt=primary_cost,
                    quantity=qty,
                    subtotal_bdt=round(cost, 2),
                )
            ],
            details=f"Greedy single-size coverage: {qty}x {primary_name}",
        )

    # Reconstruct container counts from parent pointers
    counts: dict[int, int] = defaultdict(int)
    curr = best_v
    while curr > 0 and parent[curr] is not None:
        pv, item_idx = parent[curr]
        counts[item_idx] += 1
        curr = pv

    breakdown: list[ContainerChoice] = []
    for item_idx, qty in sorted(counts.items()):
        cap, cost, name = valid_containers[item_idx]
        breakdown.append(
            ContainerChoice(
                container_name=name,
                capacity=cap,
                unit=unit_name,
                unit_price_bdt=cost,
                quantity=qty,
                subtotal_bdt=round(cost * qty, 2),
            )
        )

    total_supplied = round(best_v / scale, 4)
    waste = round(total_supplied - target_units, 4)
    summary_parts = [f"{c.quantity}x {c.container_name}" for c in breakdown]

    return DispensingPlan(
        strategy="INDIVISIBLE_CONTAINER",
        target_units_needed=target_units,
        total_units_supplied=total_supplied,
        leftover_waste=waste,
        total_cost_bdt=round(best_cost, 2),
        unit_of_measure=unit_name,
        container_breakdown=breakdown,
        details=f"Knapsack optimum: {', '.join(summary_parts)} (waste: {waste} {unit_name})",
    )


# ==============================================================================
# 2. Divisible Solid Dispensing Solver (Tablets, Capsules, Cut Blister Strips)
# ==============================================================================

def solve_divisible_solid_dispensing(
    target_units: float,
    skus: list[PackSKU],
    unit_name: str = "tablet",
) -> DispensingPlan:
    """Compute Bangladeshi retail dispensing cost for solid blister packs.

    In Bangladesh, retail pharmacies cut blister strips allowing per-unit purchase.
    We compare cut-strip dispensing against buying whole pack(s) if bulk pack discount exists.
    """
    if target_units <= 0:
        return DispensingPlan(
            strategy="CUT_STRIP",
            target_units_needed=0.0,
            total_units_supplied=0.0,
            leftover_waste=0.0,
            total_cost_bdt=0.0,
            unit_of_measure=unit_name,
            details="Zero quantity required",
        )

    # Option 1: Cut blister strip / loose unit dispensing
    valid_eff_prices = [s.effective_per_unit_bdt for s in skus if s.effective_per_unit_bdt > 0]
    cut_unit_price = min(valid_eff_prices) if valid_eff_prices else 0.0
    cost_cut_strip = round(cut_unit_price * target_units, 2)

    # Option 2: Full pack options with pack discounts
    best_pack_cost = float("inf")
    best_pack_plan: Optional[DispensingPlan] = None

    pack_options: list[tuple[float, float, str]] = []
    for s in skus:
        if s.pack_qty and s.pack_qty > 0 and s.pack_price_bdt and s.pack_price_bdt > 0:
            pack_options.append(
                (float(s.pack_qty), float(s.pack_price_bdt), f"{s.pack_qty}'s pack")
            )

    if pack_options:
        # Solve knapsack over full pack sizes to find absolute minimum full-pack cost
        pack_plan = solve_indivisible_knapsack(
            target_units=target_units,
            containers=pack_options,
            unit_name=unit_name,
        )
        if pack_plan.total_cost_bdt < best_pack_cost:
            best_pack_cost = pack_plan.total_cost_bdt
            best_pack_plan = pack_plan

    # If full pack purchase is strictly cheaper than cut strip dispensing, use full pack
    if best_pack_plan is not None and best_pack_cost < cost_cut_strip - 1e-4:
        return DispensingPlan(
            strategy="FULL_PACK",
            target_units_needed=target_units,
            total_units_supplied=best_pack_plan.total_units_supplied,
            leftover_waste=best_pack_plan.leftover_waste,
            total_cost_bdt=best_pack_plan.total_cost_bdt,
            unit_of_measure=unit_name,
            container_breakdown=best_pack_plan.container_breakdown,
            details=(
                f"Full pack discount advantage: {best_pack_plan.details}. "
                f"Saves {cost_cut_strip - best_pack_cost:.2f} BDT vs cut strips."
            ),
        )

    # Otherwise, cut strip is cheaper or equivalent (zero waste)
    return DispensingPlan(
        strategy="CUT_STRIP",
        target_units_needed=target_units,
        total_units_supplied=target_units,
        leftover_waste=0.0,
        total_cost_bdt=cost_cut_strip,
        unit_of_measure=unit_name,
        container_breakdown=[
            ContainerChoice(
                container_name="Cut blister strip (loose units)",
                capacity=1.0,
                unit=unit_name,
                unit_price_bdt=cut_unit_price,
                quantity=int(round(target_units)),
                subtotal_bdt=cost_cut_strip,
            )
        ],
        details=f"Retail cut blister strip: {target_units:.0f} units @ BDT {cut_unit_price:.2f}/unit",
    )


# ==============================================================================
# 3. Regimen Optimizer Engine
# ==============================================================================

class BrandRecord(BaseModel):
    """Catalog record for a brand with pre-parsed pricing and ontology resolution."""

    model_config = ConfigDict(extra="forbid")

    brand_id: str
    brand_name: str
    generic: str
    dosage_form: str
    strength: Optional[str] = None
    manufacturer: str
    medicine_type: str = "allopathic"
    dispensing_category: DispensingCategory
    skus: list[PackSKU] = Field(default_factory=list)
    reconciled_volume_ml: Optional[float] = None
    status: str
    entity: ResolvedEntity


class RegimenOptimizer:
    """Debate-hardened pharmaceutical regimen solver and cost optimizer."""

    def __init__(
        self,
        catalog_df: pl.DataFrame | None = None,
        parquet_path: Path | str = "data/raw/medicines.parquet",
    ) -> None:
        """Initialize the optimizer and pre-index equivalence groups."""
        if catalog_df is None:
            p = Path(parquet_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Medicines dataset not found at {p}. Fetch dataset first."
                )
            catalog_df = pl.read_parquet(p)

        self._catalog_df = catalog_df
        self.brand_lookup: dict[str, BrandRecord] = {}
        self.name_lookup: dict[str, str] = {}  # lower(brand_name) -> brand_id
        self.equiv_key_to_brands: dict[PharmaceuticalEquivalenceKey, list[str]] = defaultdict(list)

        self._index_catalog()

    def _index_catalog(self) -> None:
        """Parse packaging and resolve ontology for all catalog records."""
        parsed_results = parse_dataframe(self._catalog_df)

        for row, res in zip(self._catalog_df.iter_rows(named=True), parsed_results):
            bid = str(row["brand id"]).strip()
            bname = str(row["brand name"]).strip()
            generic = str(row["generic"]).strip()
            dosage = str(row["dosage form"]).strip()
            strength = row.get("strength")
            mfg = str(row["manufacturer"]).strip()
            mtype = str(row.get("type", "allopathic")).strip()

            entity = resolve_medicine_entity(
                brand_id=bid,
                brand_name=bname,
                generic=generic,
                dosage_form=dosage,
                strength=strength,
                medicine_type=mtype,
            )

            rec = BrandRecord(
                brand_id=bid,
                brand_name=bname,
                generic=generic,
                dosage_form=dosage,
                strength=strength,
                manufacturer=mfg,
                medicine_type=mtype,
                dispensing_category=res.dispensing_category,
                skus=res.skus,
                reconciled_volume_ml=res.reconciled_total_volume_ml,
                status=res.status,
                entity=entity,
            )

            self.brand_lookup[bid] = rec
            self.name_lookup[bname.lower()] = bid

            # Index only validly parsed brands into equivalence groups
            if res.status == "PARSED_VALID" and res.skus and entity.equivalence_key is not None:
                self.equiv_key_to_brands[entity.equivalence_key].append(bid)

    def find_brand(self, brand_query: str | int) -> BrandRecord:
        """Retrieve brand record by ID or exact/fuzzy brand name."""
        q = str(brand_query).strip()
        if q in self.brand_lookup:
            return self.brand_lookup[q]

        q_lower = q.lower()
        if q_lower in self.name_lookup:
            return self.brand_lookup[self.name_lookup[q_lower]]

        # RapidFuzz fallback for minor spelling variations
        from rapidfuzz import process, fuzz

        best = process.extractOne(
            q_lower,
            list(self.name_lookup.keys()),
            scorer=fuzz.ratio,
        )
        if best and best[1] >= 85:
            matched_id = self.name_lookup[best[0]]
            return self.brand_lookup[matched_id]

        raise KeyError(f"Brand '{brand_query}' not found in catalog.")

    def compute_brand_dispensing_cost(
        self,
        brand_id: str | int,
        target_units: float,
    ) -> DispensingPlan:
        """Compute the retail dispensing cost and leftover waste for a brand."""
        rec = self.find_brand(brand_id)
        cat = rec.dispensing_category
        skus = rec.skus

        if not skus:
            return DispensingPlan(
                strategy="UNIT_FALLBACK",
                target_units_needed=target_units,
                total_units_supplied=target_units,
                leftover_waste=0.0,
                total_cost_bdt=0.0,
                unit_of_measure="unit",
                details="No pricing SKUs available",
            )

        if cat == DispensingCategory.DIVISIBLE_SOLID:
            return solve_divisible_solid_dispensing(
                target_units=target_units,
                skus=skus,
                unit_name="tablet" if "tablet" in rec.dosage_form.lower() else "capsule",
            )

        elif cat == DispensingCategory.INDIVISIBLE_CONTAINER:
            # Build container options from parsed SKUs
            containers: list[tuple[float, float, str]] = []
            for s in skus:
                cap = s.measurement_value
                unit = s.measurement_unit or "ml"
                if cap is None and rec.reconciled_volume_ml is not None:
                    cap = rec.reconciled_volume_ml
                if cap is None:
                    cap = 1.0  # e.g., 1 inhaler or 1 bottle
                cost = s.pack_price_bdt or s.effective_per_unit_bdt
                name = f"{cap:g} {unit} {s.container_type or 'container'}"
                containers.append((float(cap), float(cost), name))

            unit_name = skus[0].measurement_unit or "ml"
            return solve_indivisible_knapsack(
                target_units=target_units,
                containers=containers,
                unit_name=unit_name,
            )

        else:
            # AMBIGUOUS_FORM: check if any container measurement exists
            has_measurement = any(s.measurement_value is not None for s in skus)
            if has_measurement:
                containers = []
                for s in skus:
                    cap = s.measurement_value or 1.0
                    cost = s.pack_price_bdt or s.effective_per_unit_bdt
                    name = f"{cap:g} {s.measurement_unit or 'unit'} container"
                    containers.append((float(cap), float(cost), name))
                return solve_indivisible_knapsack(
                    target_units=target_units,
                    containers=containers,
                    unit_name=skus[0].measurement_unit or "unit",
                )
            else:
                return solve_divisible_solid_dispensing(
                    target_units=target_units,
                    skus=skus,
                    unit_name="unit",
                )

    def optimize_regimen(
        self,
        prescription: list[PrescriptionItem | dict[str, Any]],
    ) -> RegimenOptimizationResult:
        """Run polypharmacy safety audit and compute cost-optimal brand substitutions across a regimen."""
        items: list[PrescriptionItem] = []
        for p in prescription:
            if isinstance(p, PrescriptionItem):
                items.append(p)
            elif isinstance(p, dict):
                items.append(PrescriptionItem(**p))
            else:
                raise TypeError(f"Invalid prescription item type: {type(p)}")

        if not items:
            return RegimenOptimizationResult(
                prescribed_total_cost_bdt=0.0,
                optimized_total_cost_bdt=0.0,
                total_savings_bdt=0.0,
                total_savings_pct=0.0,
                safety_alerts=[],
            )

        # 1. Polypharmacy Safety Audit
        resolved_entities: list[ResolvedEntity] = []
        for item in items:
            rec = self.find_brand(item.brand_id)
            resolved_entities.append(rec.entity)

        regimen_alerts = check_prescription_safety(resolved_entities)
        has_dup = any(a.alert_type == "DUPLICATE_ACTIVE_INGREDIENT" for a in regimen_alerts)
        has_nti = any(e.nti_locked for e in resolved_entities)

        # 2. Optimize Each Prescription Item
        item_results: list[ItemOptimizationResult] = []
        total_prescribed_cost = 0.0
        total_optimized_cost = 0.0

        for item, entity in zip(items, resolved_entities):
            rec = self.find_brand(item.brand_id)
            units_needed = item.total_units_needed

            # Prescribed brand dispensing economics
            prescribed_plan = self.compute_brand_dispensing_cost(rec.brand_id, units_needed)
            prescribed_cost = prescribed_plan.total_cost_bdt
            prescribed_waste = prescribed_plan.leftover_waste

            # Check substitution permissibility
            is_substitutable = True
            block_reason: Optional[str] = None

            if entity.nti_locked:
                is_substitutable = False
                block_reason = f"SAFETY LOCK: Narrow Therapeutic Index (NTI). {entity.nti_citation}"
            elif not entity.is_substitutable:
                is_substitutable = False
                block_reason = entity.substitution_block_reason
            elif entity.equivalence_key is None:
                is_substitutable = False
                block_reason = "No valid pharmaceutical bioequivalence key established."

            # Evaluate candidate brands within the exact equivalence group
            candidate_summaries: list[BrandCandidate] = []
            if is_substitutable and entity.equivalence_key is not None:
                peer_bids = self.equiv_key_to_brands.get(entity.equivalence_key, [])
                # Ensure prescribed brand is evaluated if valid
                all_bids = list(dict.fromkeys([rec.brand_id] + peer_bids))

                for cand_id in all_bids:
                    cand_rec = self.find_brand(cand_id)
                    cand_plan = self.compute_brand_dispensing_cost(cand_id, units_needed)
                    candidate_summaries.append(
                        BrandCandidate(
                            brand_id=cand_rec.brand_id,
                            brand_name=cand_rec.brand_name,
                            manufacturer=cand_rec.manufacturer,
                            dosage_form=cand_rec.dosage_form,
                            dispensing_category=cand_rec.dispensing_category,
                            total_cost_bdt=cand_plan.total_cost_bdt,
                            leftover_waste=cand_plan.leftover_waste,
                            dispensing_plan=cand_plan,
                        )
                    )

                # Sort candidates by cost ascending, then waste ascending
                candidate_summaries.sort(key=lambda c: (c.total_cost_bdt, c.leftover_waste))
            else:
                candidate_summaries.append(
                    BrandCandidate(
                        brand_id=rec.brand_id,
                        brand_name=rec.brand_name,
                        manufacturer=rec.manufacturer,
                        dosage_form=rec.dosage_form,
                        dispensing_category=rec.dispensing_category,
                        total_cost_bdt=prescribed_cost,
                        leftover_waste=prescribed_waste,
                        dispensing_plan=prescribed_plan,
                    )
                )

            # Determine optimal brand
            best_cand = candidate_summaries[0]
            savings_bdt = round(max(0.0, prescribed_cost - best_cand.total_cost_bdt), 2)
            savings_pct = round((savings_bdt / prescribed_cost * 100), 2) if prescribed_cost > 0 else 0.0

            total_prescribed_cost += prescribed_cost
            total_optimized_cost += best_cand.total_cost_bdt

            item_results.append(
                ItemOptimizationResult(
                    prescribed_item=item,
                    brand_id=rec.brand_id,
                    brand_name=rec.brand_name,
                    generic_name=rec.generic,
                    dosage_form=rec.dosage_form,
                    strength=rec.strength,
                    dispensing_category=rec.dispensing_category,
                    total_units_needed=units_needed,
                    is_nti_locked=entity.nti_locked,
                    is_substitutable=is_substitutable,
                    substitution_block_reason=block_reason,
                    prescribed_cost_bdt=prescribed_cost,
                    prescribed_waste=prescribed_waste,
                    prescribed_plan=prescribed_plan,
                    optimal_brand_id=best_cand.brand_id,
                    optimal_brand_name=best_cand.brand_name,
                    optimal_manufacturer=best_cand.manufacturer,
                    optimal_cost_bdt=best_cand.total_cost_bdt,
                    optimal_waste=best_cand.leftover_waste,
                    optimal_plan=best_cand.dispensing_plan,
                    savings_bdt=savings_bdt,
                    savings_pct=savings_pct,
                    num_candidates=len(candidate_summaries),
                    candidates=candidate_summaries,
                )
            )

        regimen_savings_bdt = round(max(0.0, total_prescribed_cost - total_optimized_cost), 2)
        regimen_savings_pct = (
            round((regimen_savings_bdt / total_prescribed_cost * 100), 2)
            if total_prescribed_cost > 0
            else 0.0
        )

        return RegimenOptimizationResult(
            prescribed_total_cost_bdt=round(total_prescribed_cost, 2),
            optimized_total_cost_bdt=round(total_optimized_cost, 2),
            total_savings_bdt=regimen_savings_bdt,
            total_savings_pct=regimen_savings_pct,
            safety_alerts=regimen_alerts,
            has_duplicate_ingredients=has_dup,
            has_nti_lock=has_nti,
            items=item_results,
        )


def optimize_prescription_regimen(
    prescription: list[PrescriptionItem | dict[str, Any]],
    catalog_df: pl.DataFrame | None = None,
) -> RegimenOptimizationResult:
    """Convenience function to solve a prescription regimen."""
    optimizer = RegimenOptimizer(catalog_df=catalog_df)
    return optimizer.optimize_regimen(prescription)
