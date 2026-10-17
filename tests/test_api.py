"""Integration tests for FastAPI endpoints: health, brand search, regimen optimization, and AST parser inspection."""

from fastapi.testclient import TestClient
import pytest
from src.openrx_bd.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /v1/health returns valid manifest metadata, split lock status, and benchmark summary."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "HEALTHY"
    assert data["service"] == "OpenRx-BD Clinical Cost-Optimization Engine"
    assert data["manifest"]["dataset_name"] == "Mahadih534/all-Bangladeshi-medicines"
    assert data["manifest"]["row_count"] == 21714
    assert data["manifest"]["column_count"] == 10
    assert data["cryptographic_split_lock"]["status"] == "LOCKED"
    assert "blind_test_sha256" in data["cryptographic_split_lock"]


def test_brands_search_prefix():
    """Verify /v1/brands/search supports prefix matching and returns ontology metadata."""
    response = client.get("/v1/brands/search?q=Napa&limit=10")
    assert response.status_code == 200
    results = response.json()

    assert len(results) > 0
    assert len(results) <= 10

    # Napa is a widely prescribed Paracetamol brand in Bangladesh
    first = results[0]
    assert "Napa" in first["brand_name"]
    assert "Paracetamol" in first["generic"]
    assert "brand_id" in first
    assert "dosage_form" in first
    assert "dispensing_category" in first
    assert first["canonical_generic"] == "paracetamol"
    assert first["is_nti"] is False


def test_brands_search_nti_status():
    """Verify /v1/brands/search flags Narrow Therapeutic Index brands."""
    response = client.get("/v1/brands/search?q=Unifarin")
    assert response.status_code == 200
    results = response.json()

    assert len(results) > 0
    match = next((r for r in results if r["brand_name"] == "Unifarin"), None)
    assert match is not None
    assert match["is_nti"] is True
    assert match["canonical_generic"] == "warfarin"
    assert match["salt_form"] == "sodium"
    assert match["nti_citation"] is not None


def test_regimen_optimize_polypharmacy_duplicate_active_ingredient():
    """Verify /v1/regimen/optimize detects concurrent duplicate active ingredients across brands."""
    payload = {
        "items": [
            {"brand_id": "Napa", "units_per_dose": 1.0, "doses_per_day": 3.0, "duration_days": 7},
            {"brand_id": "Ace", "units_per_dose": 1.0, "doses_per_day": 2.0, "duration_days": 5},
        ]
    }
    response = client.post("/v1/regimen/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_duplicate_ingredients"] is True
    assert len(data["safety_alerts"]) > 0

    dup_alert = next(
        (a for a in data["safety_alerts"] if a["alert_type"] == "DUPLICATE_ACTIVE_INGREDIENT"),
        None,
    )
    assert dup_alert is not None
    assert dup_alert["severity"] == "HIGH"
    assert "paracetamol" in dup_alert["active_ingredient"].lower()
    assert set(dup_alert["affected_brands"]) == {"Napa", "Ace"}


def test_regimen_optimize_nti_lock():
    """Verify /v1/regimen/optimize enforces strict substitution lock on NTI drugs."""
    payload = {
        "items": [
            {"brand_id": "Unifarin", "units_per_dose": 1.0, "doses_per_day": 1.0, "duration_days": 30},
        ]
    }
    response = client.post("/v1/regimen/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["has_nti_lock"] is True
    item = data["items"][0]
    assert item["is_nti_locked"] is True
    assert item["is_substitutable"] is False
    assert item["optimal_brand_name"] == "Unifarin"
    assert item["savings_bdt"] == 0.0


def test_regimen_optimize_indivisible_knapsack():
    """Verify /v1/regimen/optimize runs unbounded knapsack over indivisible liquid syrup containers."""
    payload = {
        "items": [
            # Safi syrup: 10 ml twice daily for 30 days = 600 ml needed
            {"brand_id": "Safi", "units_per_dose": 10.0, "doses_per_day": 2.0, "duration_days": 30},
        ]
    }
    response = client.post("/v1/regimen/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()

    item = data["items"][0]
    assert item["dispensing_category"] == "INDIVISIBLE_CONTAINER"
    plan = item["optimal_plan"]
    assert plan["strategy"] == "INDIVISIBLE_CONTAINER"
    assert plan["total_units_supplied"] >= 600.0
    assert plan["total_cost_bdt"] > 0.0


def test_parser_inspect_sandbox():
    """Verify /v1/parser/inspect decomposes packaging strings into AST chunks with invariant verification."""
    payload = {
        "package_container": "500 mg vial: ৳ 28.43,(5's pack: ৳ 142.15)",
        "dosage_form": "Injection",
        "strength": "500 mg",
    }
    response = client.post("/v1/parser/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "PARSED_VALID"
    assert data["dispensing_category"] == "INDIVISIBLE_CONTAINER"
    assert len(data["ast_tokens"]) == 1

    token = data["ast_tokens"][0]
    assert token["unit_price_bdt"] == 28.43
    assert token["pack_qty"] == 5
    assert token["pack_price_bdt"] == 142.15
    assert token["invariant_checked"] is True
    assert token["invariant_passed"] is True


def test_root_dashboard_serves_html():
    """Verify GET / serves the interactive static web UI dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "OpenRx-BD" in response.text
    assert "Regimen Optimizer" in response.text
    assert "AST Inspector" in response.text
