"""
Phase 1 Smoke Test — FastAPI Backend Health & Metadata Verification
Tests backend application creation, health check endpoint, case study metadata,
and mandatory geospatial attributions.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def check_backend_health() -> Dict[str, Any]:
    results = {}

    # 1. Import backend application
    try:
        from backend.main import app, health_check
        results["backend_import"] = "PASS"
    except ImportError as e:
        results["backend_import"] = f"FAIL: {e}"
        return results

    # 2. Test direct health check function call
    try:
        start = time.perf_counter()
        resp = health_check()
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        results["endpoint_latency_ms"] = latency_ms

        # Validate response model
        resp_dict = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
        results["status"] = resp_dict.get("status")
        results["service"] = resp_dict.get("service")
        results["version"] = resp_dict.get("version")

        # Validate case study metadata
        case_study = resp_dict.get("case_study", {})
        results["case_study_name"] = case_study.get("name")
        results["case_study_event_date"] = case_study.get("event_date")
        results["case_study_activation"] = case_study.get("copernicus_activation")

        # Validate attributions (Critical requirement for Copernicus & OSM license compliance)
        attribs = resp_dict.get("attribution", {})
        required_attrib_keys = ["sentinel", "world_dem", "osm", "model"]
        missing_attribs = [k for k in required_attrib_keys if k not in attribs or not attribs[k]]
        results["attribution_keys_present"] = len(missing_attribs) == 0

        has_valid_status = resp_dict.get("status") == "healthy"
        has_case_study = "Trishuli" in str(case_study.get("name", ""))
        results["backend_test"] = "PASS" if (has_valid_status and has_case_study and not missing_attribs) else "FAIL"

    except Exception as e:
        results["health_check_execution"] = f"FAIL: {e}"
        results["backend_test"] = "FAIL"

    return results


def test_backend():
    res = check_backend_health()
    assert res.get("backend_test") == "PASS", f"Backend test failed: {res}"


def test_api_endpoints():
    """Tests FastAPI endpoints using TestClient against actual run artifacts."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # 1. Test /health
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "Trishuli" in data["case_study"]["name"]

    # 2. Check for completed run in runs/
    runs_dir = WORKSPACE_ROOT / "runs"
    completed_runs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "manifest.json").is_file()] if runs_dir.is_dir() else []

    if completed_runs:
        sample_run_id = completed_runs[0].name

        # 3. Test GET /api/analyses/{id}
        resp_status = client.get(f"/api/analyses/{sample_run_id}")
        assert resp_status.status_code == 200
        run_data = resp_status.json()
        assert run_data["run_id"] == sample_run_id
        assert run_data["status"] == "COMPLETED"

        # 4. Test GET /api/analyses/{id}/results
        resp_res = client.get(f"/api/analyses/{sample_run_id}/results")
        assert resp_res.status_code == 200
        manifest = resp_res.json()
        assert "summary_statistics" in manifest
        assert "satellite_scenes" in manifest

        # 5. Test GET /api/analyses/{id}/geojson
        resp_gj = client.get(f"/api/analyses/{sample_run_id}/geojson?layer=snunet_flood_extent")
        assert resp_gj.status_code == 200
        gj = resp_gj.json()
        assert gj["type"] == "FeatureCollection"

        # 6. Test Phase 3 Connectivity & Derived Layers (if run has derived outputs)
        derived_summary = runs_dir / sample_run_id / "derived" / "connectivity_summary.json"
        if derived_summary.is_file():
            resp_conn = client.get(f"/api/analyses/{sample_run_id}/connectivity")
            assert resp_conn.status_code == 200
            conn_data = resp_conn.json()
            assert "settlement_accessibility" in conn_data
            assert "graph_metrics" in conn_data

            # Test settlement accessibility geojson
            resp_settle = client.get(f"/api/analyses/{sample_run_id}/geojson?layer=settlement_accessibility")
            assert resp_settle.status_code == 200
            assert resp_settle.json()["type"] == "FeatureCollection"

            # Test potentially affected roads geojson
            resp_roads = client.get(f"/api/analyses/{sample_run_id}/geojson?layer=potentially_affected_roads")
            assert resp_roads.status_code == 200
            assert resp_roads.json()["type"] == "FeatureCollection"

    # 7. Test 404 for non-existent run
    resp_404 = client.get("/api/analyses/invalid_run_999999")
    assert resp_404.status_code == 404


if __name__ == "__main__":
    print("=" * 60)
    print("FASTAPI BACKEND SMOKE TEST")
    print("=" * 60)
    res = check_backend_health()
    for k, v in res.items():
        print(f"  {k}: {v}")
    overall = "PASS" if res.get("backend_test") == "PASS" else "FAIL"
    print(f"\n  OVERALL: {overall}")
    sys.exit(0 if overall == "PASS" else 1)
