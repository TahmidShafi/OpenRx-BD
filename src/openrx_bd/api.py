"""FastAPI service providing clinical prescription optimization, polypharmacy audit, and live parser inspection."""

from contextlib import asynccontextmanager
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz, process

from src.ontology.entity_resolver import ResolvedEntity
from src.optimizer.regimen_solver import (
    PrescriptionItem,
    RegimenOptimizationResult,
    RegimenOptimizer,
)
from src.parser.container_parser import (
    CONTAINER_SPLIT_PATTERN,
    PATTERN_COMPOSITE,
    PATTERN_SINGLE_CONTAINER,
    PATTERN_STANDALONE_PACK,
    PATTERN_STANDALONE_UNIT,
    classify_dispensing_category,
    parse_container,
)
from src.parser.schemas import DispensingCategory, RowParseResult

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MANIFEST_PATH = Path("data/raw/manifest.json")
BENCHMARK_PATH = Path("reports/benchmark_results.json")
SPLIT_LOCK_PATH = Path("reports/split_lock.json")

# Global singleton optimizer loaded once on application startup
_optimizer: Optional[RegimenOptimizer] = None


def get_optimizer() -> RegimenOptimizer:
    """Retrieve pre-warmed RegimenOptimizer instance."""
    global _optimizer
    if _optimizer is None:
        _optimizer = RegimenOptimizer()
    return _optimizer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-warm dataset catalog index during application startup."""
    get_optimizer()
    yield


app = FastAPI(
    title="OpenRx-BD API",
    description="Prescription cost optimization and polypharmacy safety engine for Bangladesh",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local dashboards and developer tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ==============================================================================
# Pydantic Request & Response Models
# ==============================================================================

class RegimenOptimizeRequest(BaseModel):
    """Prescription payload containing one or more prescribed medications."""

    model_config = ConfigDict(extra="forbid")

    items: list[PrescriptionItem] = Field(
        ...,
        min_length=1,
        description="List of prescribed medicines with dosing schedule and duration",
    )


class ParserInspectRequest(BaseModel):
    """Raw package container text to evaluate in the AST parser sandbox."""

    model_config = ConfigDict(extra="forbid")

    package_container: str = Field(..., description="Raw MedEx package container string")
    dosage_form: Optional[str] = Field(None, description="Dosage form e.g. Tablet, Syrup, Injection")
    strength: Optional[str] = Field(None, description="Strength e.g. 500 mg, 25 mg/ml")


class AstTokenChunk(BaseModel):
    """Parsed AST token information for an individual SKU chunk."""

    chunk_index: int
    raw_text: str
    grammar_pattern_matched: str
    container_type: Optional[str] = None
    measurement_value: Optional[float] = None
    measurement_unit: Optional[str] = None
    unit_price_bdt: Optional[float] = None
    pack_qty: Optional[int] = None
    pack_price_bdt: Optional[float] = None
    effective_per_unit_bdt: float
    invariant_checked: bool
    invariant_passed: Optional[bool] = None
    invariant_formula: Optional[str] = None


class ParserInspectResponse(BaseModel):
    """Comprehensive AST and mathematical invariant inspection result."""

    status: str
    dispensing_category: DispensingCategory
    reconciled_volume_ml: Optional[float] = None
    ast_tokens: list[AstTokenChunk]
    raw_text: str
    abstain_reason: Optional[str] = None


class BrandSearchResult(BaseModel):
    """Lightweight brand entity summary for UI autocomplete and search."""

    brand_id: str
    brand_name: str
    generic: str
    dosage_form: str
    strength: Optional[str] = None
    manufacturer: str
    medicine_type: str
    is_nti: bool
    nti_citation: Optional[str] = None
    salt_form: Optional[str] = None
    canonical_generic: str
    who_aware_category: Optional[str] = None
    dispensing_category: DispensingCategory


# ==============================================================================
# API Endpoints
# ==============================================================================

@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the static interactive web UI dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return {
            "message": "OpenRx-BD API is active. Web UI index.html not yet initialized.",
            "docs_url": "/docs",
        }
    return FileResponse(index_file)


@app.get("/v1/health")
def get_health() -> dict[str, Any]:
    """Return dataset manifest hash, row counts, and blind benchmark summary."""
    manifest_data = {}
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    benchmark_data = {}
    if BENCHMARK_PATH.exists():
        with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
            benchmark_data = json.load(f)

    split_lock = {}
    if SPLIT_LOCK_PATH.exists():
        with open(SPLIT_LOCK_PATH, "r", encoding="utf-8") as f:
            split_lock = json.load(f)

    return {
        "status": "HEALTHY",
        "service": "OpenRx-BD Clinical Cost-Optimization Engine",
        "version": "0.1.0",
        "manifest": {
            "dataset_name": manifest_data.get("dataset_name", "Mahadih534/all-Bangladeshi-medicines"),
            "sha256": manifest_data.get("sha256"),
            "row_count": manifest_data.get("row_count", 21714),
            "column_count": manifest_data.get("column_count", 10),
            "fetched_at_utc": manifest_data.get("fetched_at_utc"),
        },
        "cryptographic_split_lock": {
            "blind_test_sha256": split_lock.get("sha256"),
            "status": "LOCKED",
        },
        "benchmark_summary": benchmark_data.get("splits", {}),
        "price_dispersion": benchmark_data.get("market_price_dispersion", {}),
    }


@app.get("/v1/brands/search", response_model=list[BrandSearchResult])
def search_brands(
    q: str = Query(..., min_length=1, description="Brand name, generic, or ID prefix query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results to return"),
) -> list[BrandSearchResult]:
    """Fast prefix and fuzzy search over indexed pharmaceutical brands."""
    opt = get_optimizer()
    query = q.strip().lower()
    if not query:
        return []

    prefix_matches: list[Any] = []
    substring_matches: list[Any] = []
    seen_ids: set[str] = set()

    # Pass 1: Prefix match on brand name
    for bid, rec in opt.brand_lookup.items():
        bname_lower = rec.brand_name.lower()
        if bname_lower.startswith(query):
            prefix_matches.append(rec)
            seen_ids.add(bid)
            if len(prefix_matches) >= limit:
                break

    # Pass 2: Substring match on brand name or generic
    if len(prefix_matches) < limit:
        for bid, rec in opt.brand_lookup.items():
            if bid in seen_ids:
                continue
            if query in rec.brand_name.lower() or query in rec.generic.lower() or query == bid:
                substring_matches.append(rec)
                seen_ids.add(bid)
                if len(prefix_matches) + len(substring_matches) >= limit:
                    break

    combined = (prefix_matches + substring_matches)[:limit]

    # Pass 3: RapidFuzz fallback if fewer than 5 matches found
    if len(combined) < 5:
        brand_names = list(opt.name_lookup.keys())
        fuzzy_hits = process.extract(
            query,
            brand_names,
            scorer=fuzz.ratio,
            limit=limit,
        )
        for name_hit, score, _ in fuzzy_hits:
            if score >= 75:
                hit_id = opt.name_lookup[name_hit]
                if hit_id not in seen_ids:
                    rec = opt.brand_lookup[hit_id]
                    combined.append(rec)
                    seen_ids.add(hit_id)
                    if len(combined) >= limit:
                        break

    results: list[BrandSearchResult] = []
    for rec in combined:
        ing0 = rec.entity.ingredients[0] if rec.entity.ingredients else None
        results.append(
            BrandSearchResult(
                brand_id=rec.brand_id,
                brand_name=rec.brand_name,
                generic=rec.generic,
                dosage_form=rec.dosage_form,
                strength=rec.strength,
                manufacturer=rec.manufacturer,
                medicine_type=rec.medicine_type,
                is_nti=rec.entity.nti_locked,
                nti_citation=rec.entity.nti_citation,
                salt_form=ing0.salt_form if ing0 else None,
                canonical_generic=ing0.canonical_name if ing0 else rec.generic,
                who_aware_category=rec.entity.who_aware_category,
                dispensing_category=rec.dispensing_category,
            )
        )

    return results


@app.post("/v1/regimen/optimize", response_model=RegimenOptimizationResult)
def optimize_regimen_endpoint(payload: RegimenOptimizeRequest) -> RegimenOptimizationResult:
    """Optimize a multi-drug prescription regimen with polypharmacy safety audits and exact retail dispensing economics."""
    opt = get_optimizer()
    try:
        return opt.optimize_regimen(payload.items)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(exc)}")


@app.post("/v1/parser/inspect", response_model=ParserInspectResponse)
def inspect_parser_endpoint(payload: ParserInspectRequest) -> ParserInspectResponse:
    """Inspect AST tokens, mathematical invariants, and dispensing categories for any package container string."""
    raw_str = payload.package_container.strip()
    parse_res = parse_container(
        brand_id="sandbox",
        package_container=raw_str,
        dosage_form=payload.dosage_form,
        strength=payload.strength,
    )

    # Decompose into AST token chunks and identify matched grammar patterns
    chunks: list[str] = []
    for c in raw_str.split(",,"):
        sub = CONTAINER_SPLIT_PATTERN.split(c)
        chunks.extend([s.strip() for s in sub if s.strip()])

    ast_tokens: list[AstTokenChunk] = []
    for idx, sku in enumerate(parse_res.skus):
        chunk_text = chunks[idx] if idx < len(chunks) else raw_str

        # Grammar identification
        if PATTERN_COMPOSITE.search(chunk_text):
            pattern_id = "PATTERN_1_COMPOSITE_UNIT_AND_PACK"
        elif PATTERN_SINGLE_CONTAINER.search(chunk_text):
            pattern_id = "PATTERN_2_SINGLE_CONTAINER_SPEC"
        elif PATTERN_STANDALONE_PACK.search(chunk_text):
            pattern_id = "PATTERN_3_STANDALONE_PACK"
        elif PATTERN_STANDALONE_UNIT.search(chunk_text):
            pattern_id = "PATTERN_4_STANDALONE_UNIT"
        else:
            pattern_id = "PATTERN_UNSPECIFIED"

        formula_str = None
        if sku.invariant_checked:
            diff = abs((sku.unit_price_bdt or 0) * (sku.pack_qty or 1) - (sku.pack_price_bdt or 0))
            status_text = "PASS" if sku.invariant_passed else "FAIL"
            formula_str = (
                f"|{sku.unit_price_bdt} * {sku.pack_qty} - {sku.pack_price_bdt}| "
                f"= {diff:.4f} <= 0.05 ({status_text})"
            )

        ast_tokens.append(
            AstTokenChunk(
                chunk_index=idx,
                raw_text=chunk_text,
                grammar_pattern_matched=pattern_id,
                container_type=sku.container_type,
                measurement_value=sku.measurement_value,
                measurement_unit=sku.measurement_unit,
                unit_price_bdt=sku.unit_price_bdt,
                pack_qty=sku.pack_qty,
                pack_price_bdt=sku.pack_price_bdt,
                effective_per_unit_bdt=sku.effective_per_unit_bdt,
                invariant_checked=sku.invariant_checked,
                invariant_passed=sku.invariant_passed,
                invariant_formula=formula_str,
            )
        )

    return ParserInspectResponse(
        status=parse_res.status,
        dispensing_category=parse_res.dispensing_category,
        reconciled_volume_ml=parse_res.reconciled_total_volume_ml,
        ast_tokens=ast_tokens,
        raw_text=raw_str,
        abstain_reason=parse_res.abstain_reason,
    )


@app.get("/v1/benchmark")
def get_benchmark_report() -> dict[str, Any]:
    """Retrieve frozen benchmark results and price dispersion statistics."""
    if not BENCHMARK_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Benchmark report not found. Run run_blind_benchmark first.",
        )
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
