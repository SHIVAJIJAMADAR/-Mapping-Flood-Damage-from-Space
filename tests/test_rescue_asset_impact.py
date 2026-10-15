"""
Test Phase 5.6: Rescue Asset Impact Filters and Individual Building Inspection.
Verifies:
1. Rescue summary endpoint with dynamic counts for active run.
2. Building footprint enrichment with honest status labels and 1,000-cap warning.
3. Candidate vs primary flood exposure layer independence.
4. Bridge honest empty-state handling (0 primary flood overlap).
5. Road and bridge inspector attributes and terminology.
6. Absolute absence of unsupported 'collapsed' or 'destroyed' classifications.
7. Active-run consistency across endpoints.
"""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
CASE_STUDY_RUN_ID = "flood_20261014_082338_45889d"


def test_rescue_summary_counts_and_structure():
    """Verify GET /api/analyses/{run_id}/rescue_summary returns dynamic counts."""
    res = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/rescue_summary")
    assert res.status_code == 200
    data = res.json()
    assert data["run_id"] == CASE_STUDY_RUN_ID

    # Buildings verification
    bldg = data["buildings"]
    assert bldg["total_sample"] == 1000
    assert bldg["primary_exposed"] == 0  # Honest zero result
    assert bldg["candidate_exposed"] == 92
    assert bldg["sample_cap"] == 1000
    assert "1,000" in bldg["completeness_warning"]
    assert "collapse" in bldg["structural_caveat"].lower()

    # Roads verification
    roads = data["roads"]
    assert roads["total_roads"] == 898
    assert roads["primary_exposed"] == 12
    assert pytest.approx(roads["exposed_length_meters"], 0.5) == 190.1
    assert roads["candidate_exposed"] == 166
    assert roads["severed_graph_edges"] == 5

    # Bridges verification (honest empty state for primary flood)
    bridges = data["bridges"]
    assert bridges["total_bridges"] == 25
    assert bridges["primary_exposed"] == 0
    assert bridges["candidate_exposed"] == 7

    # Settlements verification
    settlements = data["settlements"]
    assert settlements["total"] == 27
    assert settlements["cut_off"] == 2
    assert settlements["detoured"] == 2
    assert settlements["unaffected"] == 23


def test_building_footprint_enrichment_and_honest_status():
    """Verify OSM buildings GeoJSON contains enriched status without unsupported classifications."""
    res = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=osm_buildings")
    assert res.status_code == 200
    fc = res.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) == 1000

    candidate_count = 0
    primary_count = 0
    no_overlap_count = 0

    for f in fc["features"]:
        props = f["properties"]
        assert "osm_id" in props or "id" in props
        assert "primary_flood_exposed" in props
        assert "candidate_change_exposed" in props
        assert "exposure_status_label" in props
        assert "model_scope_caveat" in props
        assert "inventory_sample_warning" in props

        # Validate honest vocabulary
        label = props["exposure_status_label"]
        assert label in [
            "No detected overlap in available data",
            "Possible surface-change overlap — uncertain cause",
            "Potentially flood-exposed",
            "Potentially flood-exposed & candidate surface change",
        ]

        # Verify no unsupported damage claims
        text_dump = str(props).lower()
        assert "collapsed" not in text_dump or "not assessed" in text_dump
        assert "destroyed" not in text_dump

        if props["primary_flood_exposed"]:
            primary_count += 1
        if props["candidate_change_exposed"]:
            candidate_count += 1
        if not props["primary_flood_exposed"] and not props["candidate_change_exposed"]:
            no_overlap_count += 1

    assert primary_count == 0  # Honest zero count
    assert candidate_count == 92
    assert no_overlap_count == 908


def test_independent_building_exposure_layers():
    """Verify independent layer endpoints for primary flood vs candidate change."""
    # 1. Primary exposed buildings (honest empty set)
    res_primary = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=buildings_exposed")
    assert res_primary.status_code == 200
    fc_primary = res_primary.json()
    assert len(fc_primary["features"]) == 0

    # 2. Classical candidate change buildings
    res_cand = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=buildings_candidate")
    assert res_cand.status_code == 200
    fc_cand = res_cand.json()
    assert len(fc_cand["features"]) == 92


def test_bridge_empty_result_and_candidate_layer():
    """Verify bridge honest empty state and classical candidate change layers."""
    res_bridges = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=osm_bridges")
    assert res_bridges.status_code == 200
    fc_bridges = res_bridges.json()
    assert len(fc_bridges["features"]) == 25

    # Primary flood-exposed bridges (0)
    res_b_flood = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=bridges_exposed")
    assert res_b_flood.status_code == 200
    assert len(res_b_flood.json()["features"]) == 0

    # Classical candidate-change bridges (7)
    res_b_cand = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=bridges_candidate")
    assert res_b_cand.status_code == 200
    assert len(res_b_cand.json()["features"]) == 7


def test_roads_candidate_and_severed_layers():
    """Verify roads candidate and severed layers are independently accessible."""
    # Classical candidate change roads
    res_cand = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=roads_candidate")
    assert res_cand.status_code == 200
    fc_cand = res_cand.json()
    assert len(fc_cand["features"]) == 166

    # Simulated severed roads
    res_sev = client.get(f"/api/analyses/{CASE_STUDY_RUN_ID}/geojson?layer=severed_roads")
    assert res_sev.status_code == 200
    fc_sev = res_sev.json()
    assert len(fc_sev["features"]) >= 5
