"""
Phase 4 Integration Tests — Real-Time Interactive Satellite Comparison Dashboard
Validates:
1. Raster-tile endpoint layer allowlisting and invalid layer rejection.
2. Tile coordinate bounds and negative coordinate rejection.
3. Path traversal protection on run IDs.
4. Real raster tile delivery (256x256 image/png) from the genuine 2026 run.
5. Point inspection coordinate transformation, sampling, and deterministic scientific explanation.
6. Point inspection handling of coordinates outside the satellite footprint.
7. Headline statistics and reconciled road/graph count integrity.
8. Building completeness warning propagation.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.main import app
from backend.services.tile_service import ALLOWED_LAYERS, render_raster_tile, xyz_to_epsg3857_bounds
from backend.services.inspect_service import inspect_point

client = TestClient(app)
SAMPLE_RUN_ID = "flood_20261014_082338_45889d"


def test_tile_layer_allowlist_and_validation():
    """Verify that only explicitly permitted layers can be rendered as raster tiles."""
    expected_layers = {
        "pre_vv", "post_vv", "pre_vh", "post_vh", "delta_vv_db",
        "snunet_flood_prob", "snunet_pred_class", "baseline_candidate_flood",
        "dem", "slope"
    }
    assert expected_layers.issubset(set(ALLOWED_LAYERS.keys()))

    # Allowed layer must succeed
    resp = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/tiles/pre_vv/12/3016/1715.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert len(resp.content) > 1000

    # Disallowed layer must return 400 Bad Request
    resp_bad = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/tiles/unauthorized_layer/12/3016/1715.png")
    assert resp_bad.status_code == 400
    assert "not allowed" in resp_bad.json()["detail"]


def test_tile_coordinates_and_bounds_math():
    """Verify XYZ to Web Mercator EPSG:3857 conversion and negative coordinate rejection."""
    # Negative coordinates must return 400
    resp_neg = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/tiles/pre_vv/12/-1/1715.png")
    assert resp_neg.status_code == 400

    # Check bounds calculation at zoom 0
    min_x, min_y, max_x, max_y = xyz_to_epsg3857_bounds(0, 0, 0)
    assert round(min_x) == -20037508
    assert round(max_x) == 20037508
    assert round(min_y) == -20037508
    assert round(max_y) == 20037508


def test_path_traversal_protection():
    """Verify that malicious run_id inputs cannot escape the runs/ directory."""
    resp = client.get("/api/analyses/..%2F..%2Fetc%2Fpasswd/tiles/pre_vv/12/3016/1715.png")
    assert resp.status_code in [400, 404]

    resp_insp = client.get("/api/analyses/..%2F..%2Fetc%2Fpasswd/inspect?lon=85.2&lat=28.0")
    assert resp_insp.status_code in [400, 404]


def test_real_raster_tile_delivery():
    """Verify that genuine 2026 run delivers valid PNG tiles for comparison layers."""
    for layer in ["pre_vv", "post_vv", "delta_vv_db", "snunet_pred_class", "dem"]:
        resp = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/tiles/{layer}/12/3016/1715.png")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert len(resp.content) > 100
        assert "Cache-Control" in resp.headers


def test_point_inspection_inside_footprint():
    """Verify point inspector returns real radar dB values, terrain context, and explanation."""
    resp = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/inspect?lon=85.20&lat=28.00")
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_inside_aoi"] is True
    assert data["is_valid_data"] is True
    assert "measurements" in data
    m = data["measurements"]
    assert -30.0 <= m["pre_vv_db"] <= 5.0
    assert -30.0 <= m["post_vv_db"] <= 5.0
    assert "delta_vv_db" in m
    assert "snunet_class" in m
    assert "elevation_meters" in m
    assert m["elevation_meters"] > 0
    assert "slope_degrees" in m
    assert m["slope_degrees"] >= 0

    assert "explanation" in data
    assert len(data["explanation"]) > 20
    assert "backscatter decrease" in data["explanation"].lower()
    assert "slope filter" in data["explanation"].lower()
    assert "snunet classification indicates no water" in data["explanation"].lower()
    assert "observation_dates" in data
    assert "2026-08-24" in data["observation_dates"]["pre_datetime"]
    assert "2026-09-05" in data["observation_dates"]["post_datetime"]


def test_categorical_tile_resampling_nearest():
    """Verify categorical tiles use nearest-neighbour resampling and produce discrete palette classes without intermediate blend artifacts."""
    from backend.services.tile_service import render_raster_tile
    from PIL import Image
    import io
    import numpy as np

    # Test categorical layer snunet_pred_class at zoom 13 inside the AOI
    png_bytes = render_raster_tile(SAMPLE_RUN_ID, "snunet_pred_class", 13, 6034, 3431)
    img = Image.open(io.BytesIO(png_bytes))
    arr = np.array(img)
    assert arr.shape == (256, 256, 4)

    # In snunet_pred_class:
    # Class 0: (0, 0, 0, 0)
    # Class 1: (6, 182, 212, 180)
    # Class 2: (59, 130, 246, 220)
    # Any pixel with alpha > 0 must match either Class 1 or Class 2 RGB palette exactly (no bilinear blend bleed)
    non_zero_alpha = arr[arr[:, :, 3] > 0]
    for px in non_zero_alpha:
        is_class1 = (px[0] == 6 and px[1] == 182 and px[2] == 212)
        is_class2 = (px[0] == 59 and px[1] == 130 and px[2] == 246)
        assert is_class1 or is_class2, f"Unexpected blended color found in categorical tile: {px}"


def test_point_inspection_outside_footprint():
    """Verify point inspector explicitly handles coordinates outside the satellite imagery AOI."""
    resp = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/inspect?lon=80.00&lat=25.00")
    assert resp.status_code == 200
    data = resp.json()

    assert data["is_inside_aoi"] is False
    assert data["is_valid_data"] is False
    assert "outside" in data["explanation"].lower()


def test_manifest_reconciled_integrity():
    """Verify headline statistics exclude processing padding and clarify road vs graph counts."""
    resp = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/results")
    assert resp.status_code == 200
    manifest = resp.json()

    summary = manifest["summary_statistics"]
    assert "analysis_footprint" in summary
    footprint = summary["analysis_footprint"]
    assert footprint["user_requested_aoi_area_sqkm"] < footprint["padded_processing_grid_area_sqkm"]
    assert footprint["processing_padding_meters"] == 200.0

    # Building inventory completeness warning
    infra = manifest["infrastructure_impact"]
    assert "building_inventory_audit" in infra
    bldg_audit = infra["building_inventory_audit"]
    assert bldg_audit["is_capped"] is True
    assert "capped at 1,000" in bldg_audit["completeness_warning"].lower()

    # Reconciled road vs graph edge units
    impact = infra["primary_scenario"]
    roads_info = impact["potentially_exposed_roads"]
    assert "unique_affected_osm_ways" in roads_info
    assert roads_info["unique_affected_osm_ways"] == 12

    graph_metrics = manifest["connectivity"]["primary_scenario"]["graph_metrics"]
    assert graph_metrics["simulated_severed_edges_count"] == 1436
    assert "topological graph-edge fragments" in graph_metrics["severed_edges_unit"]


def test_active_run_identity_and_listing():
    """Verify active-run labels are strictly derived from manifest event_date and run_id."""
    resp = client.get("/api/analyses")
    assert resp.status_code == 200
    runs = resp.json()
    assert len(runs) >= 2

    # Verify 2026 Case Study run
    case_study_run = next((r for r in runs if r["run_id"] == SAMPLE_RUN_ID), None)
    assert case_study_run is not None
    assert case_study_run["is_case_study"] is True
    assert "Verified Case Study" in case_study_run["label"]
    assert "2026-08-26" in case_study_run["label"]

    # Verify 2022 Historical run
    hist_2022_run = next((r for r in runs if r["run_id"] == "flood_20261014_132015_383131"), None)
    if hist_2022_run:
        assert hist_2022_run["is_case_study"] is False
        assert "Historical Analysis" in hist_2022_run["label"]
        assert "2022-09-22" in hist_2022_run["label"]
        assert "Verified Case Study" not in hist_2022_run["label"]


def test_settlement_connectivity_count_and_names_consistency():
    """Verify that displayed cut-off counts and isolated settlement names match exactly."""
    # 2026 run: 2 cut-off (Laharepauwa, Dhunge)
    resp26 = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/connectivity")
    assert resp26.status_code == 200
    conn26 = resp26.json()["settlement_accessibility"]
    assert conn26["potentially_cut_off_count"] == 2
    assert len(conn26["isolated_settlements"]) == 2
    assert "Laharepauwa" in conn26["isolated_settlements"]
    assert "Dhunge" in conn26["isolated_settlements"]

    # 2022 run: 1 cut-off (Tupche)
    resp22 = client.get("/api/analyses/flood_20261014_132015_383131/connectivity")
    assert resp22.status_code == 200
    conn22 = resp22.json()["settlement_accessibility"]
    assert conn22["potentially_cut_off_count"] == 1
    assert len(conn22["isolated_settlements"]) == 1
    assert "Tupche" in conn22["isolated_settlements"]
    assert "Laharepauwa" not in conn22["isolated_settlements"]
    assert "Dhunge" not in conn22["isolated_settlements"]


def test_individual_asset_layers_and_inspector_contents():
    """Verify individual asset GeoJSON layers deliver valid properties and warnings."""
    # OSM roads
    resp_roads = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/geojson?layer=osm_roads")
    assert resp_roads.status_code == 200
    roads_fc = resp_roads.json()
    assert len(roads_fc["features"]) > 0
    sample_road = roads_fc["features"][0]["properties"]
    assert "osm_id" in sample_road
    assert "highway" in sample_road

    # Potentially affected roads
    resp_aff = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/geojson?layer=potentially_affected_roads")
    assert resp_aff.status_code == 200
    aff_fc = resp_aff.json()
    assert len(aff_fc["features"]) == 12
    sample_aff = aff_fc["features"][0]["properties"]
    assert sample_aff["exposure_status"] == "potentially_exposed"
    assert "exposed_length_meters" in sample_aff

    # Buildings (capped at 1,000)
    resp_bldg = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/geojson?layer=osm_buildings")
    assert resp_bldg.status_code == 200
    bldg_fc = resp_bldg.json()
    assert 0 < len(bldg_fc["features"]) <= 1000

    # Bridges
    resp_br = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/geojson?layer=osm_bridges")
    assert resp_br.status_code == 200
    br_fc = resp_br.json()
    assert len(br_fc["features"]) > 0


def test_point_inspector_headline_and_plain_english_interpretation():
    """Verify point inspector returns concise responder-first headlines and disclaimers."""
    # Point inside footprint
    resp = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/inspect?lon=85.20&lat=28.00")
    assert resp.status_code == 200
    data = resp.json()
    assert "headline" in data
    assert len(data["headline"]) > 5
    assert "short_interpretation" in data
    assert len(data["short_interpretation"]) > 10
    assert "disclaimer" in data
    assert "absence of a predicted flood pixel" in data["disclaimer"].lower()

    # Point outside footprint
    resp_out = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/inspect?lon=70.00&lat=20.00")
    assert resp_out.status_code == 200
    out_data = resp_out.json()
    assert out_data["headline"] == "Outside Analyzed Footprint"
    assert "outside" in out_data["short_interpretation"].lower()


def test_dynamic_situation_report_generation_for_runs():
    """Verify dynamic A4 situation reports can be generated for active runs without date confusion."""
    # 2026 report HTML & PDF
    resp_h26 = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/report.html")
    assert resp_h26.status_code == 200
    assert "2026-08-26" in resp_h26.text
    assert "Laharepauwa" in resp_h26.text

    resp_p26 = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/report.pdf")
    assert resp_p26.status_code == 200
    assert resp_p26.headers["content-type"] == "application/pdf"
    assert len(resp_p26.content) > 10000

    # 2022 historical report HTML & PDF
    resp_h22 = client.get("/api/analyses/flood_20261014_132015_383131/report.html")
    assert resp_h22.status_code == 200
    assert "2022-09-22" in resp_h22.text
    assert "Tupche" in resp_h22.text
    assert "Laharepauwa" not in resp_h22.text

    resp_p22 = client.get("/api/analyses/flood_20261014_132015_383131/report.pdf")
    assert resp_p22.status_code == 200
    assert resp_p22.headers["content-type"] == "application/pdf"
    assert len(resp_p22.content) > 10000


def test_settlement_mutual_exclusivity_and_detoured_consistency():
    """
    Phase 5.4 Consistency Audit:
    Verify that settlement counts match settlement names for every status,
    and no settlement appears under mutually exclusive statuses in the same run.
    """
    # 1. Test 2026 Case Study Run
    resp26 = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/connectivity")
    assert resp26.status_code == 200
    data26 = resp26.json()
    conn26 = data26["settlement_accessibility"]

    cutoff26_names = set(conn26["isolated_settlements"])
    detoured26_names = set(conn26.get("detoured_settlement_names", []))

    # Verify counts match names
    assert len(cutoff26_names) == conn26["potentially_cut_off_count"] == 2
    assert len(detoured26_names) == conn26["detoured_count"] == 2

    # Verify expected settlements
    assert cutoff26_names == {"Laharepauwa", "Dhunge"}
    assert detoured26_names == {"Tupche", "Uttargaya"}

    # Verify strict mutual exclusivity (intersection must be empty)
    assert cutoff26_names.isdisjoint(detoured26_names), "Contradictory settlement status in 2026 run!"

    # 2. Test 2022 Historical Run
    resp22 = client.get("/api/analyses/flood_20261014_132015_383131/connectivity")
    assert resp22.status_code == 200
    data22 = resp22.json()
    conn22 = data22["settlement_accessibility"]

    cutoff22_names = set(conn22["isolated_settlements"])
    detoured22_names = set(conn22.get("detoured_settlement_names", []))

    # Verify counts match names
    assert len(cutoff22_names) == conn22["potentially_cut_off_count"] == 1
    assert len(detoured22_names) == conn22["detoured_count"] == 2

    # Tupche is strictly cut-off in 2022, NOT detoured!
    assert cutoff22_names == {"Tupche"}
    assert detoured22_names == {"Khalchet", "Sarsyu"}

    # Verify strict mutual exclusivity (Tupche cannot appear in both)
    assert cutoff22_names.isdisjoint(detoured22_names), "Contradictory settlement status: Tupche in both statuses!"
    assert "Tupche" not in detoured22_names


def test_temporal_osm_snapshot_enforcement():
    """
    Phase 5.4 Temporal Integrity Audit:
    Enforce that an OSM snapshot must not postdate the event date.
    Verify that 2026 run is marked valid, 2022 run is flagged as temporally invalid,
    and attempting to trigger connectivity with a future snapshot raises HTTP 400.
    """
    # 2026 run: Event 2026-08-26 with snapshot 2026-07-27 -> Valid
    resp26 = client.get(f"/api/analyses/{SAMPLE_RUN_ID}/connectivity")
    assert resp26.status_code == 200
    assert resp26.json().get("is_temporally_valid") is True

    # 2022 run: Event 2022-09-22 with snapshot 2026-07-27 -> Invalid!
    resp22 = client.get("/api/analyses/flood_20261014_132015_383131/connectivity")
    assert resp22.status_code == 200
    data22 = resp22.json()
    assert data22.get("is_temporally_valid") is False
    assert "temporal_warning" in data22
    assert "postdates event date" in data22["temporal_warning"]

    # Reject future snapshot timestamp with HTTP 400
    resp_bad = client.post(
        f"/api/analyses/{SAMPLE_RUN_ID}/connectivity?snapshot_timestamp=2030-01-01T00:00:00Z"
    )
    assert resp_bad.status_code == 400
    assert "cannot postdate event date" in resp_bad.json()["detail"]


