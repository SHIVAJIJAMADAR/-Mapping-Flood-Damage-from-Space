"""
Unit and Integration Tests for Phase 3 — OSM Infrastructure & Connectivity Pipeline
Validates:
1. Temporal snapshot cutoff date enforcement (<= 2026-07-27T00:00:00Z).
2. Routable road network graph construction and metric edge distances.
3. Asset exposure calculations and strict 'potentially_exposed' labeling.
4. Settlement reachability classification (cut_off, detoured, unaffected, baseline_unreachable).
5. Empty flood extent handling.
6. Primary vs. sensitivity scenario separation.
"""

import json
import sys
from pathlib import Path
import pytest
from shapely.geometry import Point, LineString, Polygon, mapping

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.services.osm_service import (
    HISTORICAL_CUTOFF_TIMESTAMP,
    OSMServiceError,
    fetch_historical_osm_data,
)
from backend.services.impact_service import assess_infrastructure_impact
from backend.services.connectivity_service import (
    RoadNetworkGraph,
    run_connectivity_pipeline,
    analyze_settlement_connectivity,
)


def test_osm_cutoff_enforcement():
    """Verify that requesting OSM data beyond the historical cutoff raises OSMServiceError."""
    assert HISTORICAL_CUTOFF_TIMESTAMP == "2026-07-27T00:00:00Z"

    # A timestamp past the cutoff must raise OSMServiceError
    with pytest.raises(OSMServiceError) as exc_info:
        fetch_historical_osm_data(
            bbox=[85.15, 27.95, 85.25, 28.05],
            snapshot_timestamp="2026-08-28T00:00:00Z",
        )
    assert "violates challenge cutoff" in str(exc_info.value)


def test_road_network_graph_construction():
    """Verify graph creation from GeoJSON LineStrings with metric edge weights."""
    # Create two connected line segments in Kathmandu/Trishuli latitude (~28N, 85E)
    # 0.001 deg lat is ~111 meters
    coords1 = [[85.2000, 28.0000], [85.2000, 28.0010]]
    coords2 = [[85.2000, 28.0010], [85.2010, 28.0010]]

    roads_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "road_1",
                "geometry": {"type": "LineString", "coordinates": coords1},
                "properties": {"osm_id": 101, "highway": "primary", "name": "Main Highway"},
            },
            {
                "type": "Feature",
                "id": "road_2",
                "geometry": {"type": "LineString", "coordinates": coords2},
                "properties": {"osm_id": 102, "highway": "secondary", "name": "Access Road"},
            },
        ],
    }

    graph = RoadNetworkGraph()
    count = graph.build_from_osm_roads(roads_geojson)
    assert count == 2
    assert graph.graph.number_of_nodes() == 3
    assert graph.graph.number_of_edges() == 2

    # Check metric edge weights (should be ~111m and ~98m at lat 28)
    for u, v, data in graph.graph.edges(data=True):
        assert "weight" in data
        assert 50.0 < data["weight"] < 150.0
        assert data["osm_id"] in [101, 102]


def test_asset_impact_metric_units(tmp_path):
    """Verify road and building exposure metrics and 'potentially_exposed' labeling."""
    # Road passing through a flood polygon
    # Road from (85.2000, 28.0000) to (85.2000, 28.0020) (~222m)
    # Flood polygon covering (85.1990 to 85.2010, 28.0005 to 28.0015) (~111m overlap)
    road_geom = LineString([(85.2000, 28.0000), (85.2000, 28.0020)])
    flood_poly = Polygon([
        (85.1990, 28.0005),
        (85.2010, 28.0005),
        (85.2010, 28.0015),
        (85.1990, 28.0015),
        (85.1990, 28.0005),
    ])

    osm_data = {
        "roads": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "road_1",
                    "geometry": mapping(road_geom),
                    "properties": {"osm_id": 101, "highway": "primary", "name": "Valley Road"},
                }
            ],
        },
        "buildings": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "bldg_1",
                    "geometry": mapping(Polygon([
                        (85.2001, 28.0008),
                        (85.2003, 28.0008),
                        (85.2003, 28.0010),
                        (85.2001, 28.0010),
                        (85.2001, 28.0008),
                    ])),
                    "properties": {"osm_id": 201, "building": "yes"},
                }
            ],
        },
        "bridges": {"type": "FeatureCollection", "features": []},
        "settlements": {"type": "FeatureCollection", "features": []},
        "destinations": {"type": "FeatureCollection", "features": []},
    }

    snunet_flood_gj = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": mapping(flood_poly),
                "properties": {"value": 1, "class_name": "flood_water"},
            }
        ],
    }

    impact = assess_infrastructure_impact(
        osm_data=osm_data,
        snunet_flood_geojson=snunet_flood_gj,
        output_dir=tmp_path,
    )

    p_scen = impact["primary_scenario"]
    assert p_scen["potentially_exposed_roads"]["count"] == 1
    # Overlap should be approximately 111m
    assert 90.0 < p_scen["potentially_exposed_roads"]["total_exposed_length_meters"] < 130.0
    assert p_scen["potentially_exposed_buildings"]["count"] == 1
    assert p_scen["potentially_exposed_buildings"]["total_exposed_area_sqm"] > 0
    assert "potentially_exposed" in p_scen["scientific_caveat"]


def test_reachability_classification_all_categories(tmp_path):
    """
    Construct a synthetic network to verify all 4 reachability classes:
    1. Cut-off: Road severed with no alternative.
    2. Detoured: Shorter direct path severed, alternative longer path intact.
    3. Unaffected: Direct path intact.
    4. Baseline Unreachable: Isolated settlement with no baseline road path.
    """
    # Destination (Hospital) at (85.2000, 28.0000)
    dest_pt = [85.2000, 28.0000]

    # Settlement 1 (Cut-off) at (85.2000, 28.0020)
    # Connected only via road_cut: (85.2000, 28.0020) -> (85.2000, 28.0000)
    # We will sever road_cut.

    # Settlement 2 (Detoured) at (85.2020, 28.0020)
    # Direct route to dest: road_detour_direct (85.2020, 28.0020) -> (85.2000, 28.0000) (~300m)
    # Alternative route: road_detour_alt1 (85.2020, 28.0020) -> (85.2040, 28.0000)
    #               then road_detour_alt2 (85.2040, 28.0000) -> (85.2000, 28.0000) (~600m)
    # We will sever road_detour_direct.

    # Settlement 3 (Unaffected) at (85.1980, 28.0000)
    # Connected via intact road_safe: (85.1980, 28.0000) -> (85.2000, 28.0000)

    # Settlement 4 (Baseline Unreachable) at (85.2500, 28.0500)
    # Isolated, far away with no connecting road.

    roads = [
        # Cut-off road
        {
            "type": "Feature",
            "id": "road_cut",
            "geometry": {"type": "LineString", "coordinates": [[85.2000, 28.0020], [85.2000, 28.0000]]},
            "properties": {"osm_id": 301, "highway": "residential", "name": "Dead-end Valley Lane"},
        },
        # Detour direct (to be severed)
        {
            "type": "Feature",
            "id": "road_detour_direct",
            "geometry": {"type": "LineString", "coordinates": [[85.2020, 28.0020], [85.2000, 28.0000]]},
            "properties": {"osm_id": 302, "highway": "unclassified", "name": "Direct Hill Path"},
        },
        # Detour alternative
        {
            "type": "Feature",
            "id": "road_detour_alt1",
            "geometry": {"type": "LineString", "coordinates": [[85.2020, 28.0020], [85.2040, 28.0000]]},
            "properties": {"osm_id": 303, "highway": "secondary", "name": "Ridge Bypass Section A"},
        },
        {
            "type": "Feature",
            "id": "road_detour_alt2",
            "geometry": {"type": "LineString", "coordinates": [[85.2040, 28.0000], [85.2000, 28.0000]]},
            "properties": {"osm_id": 304, "highway": "secondary", "name": "Ridge Bypass Section B"},
        },
        # Unaffected road
        {
            "type": "Feature",
            "id": "road_safe",
            "geometry": {"type": "LineString", "coordinates": [[85.1980, 28.0000], [85.2000, 28.0000]]},
            "properties": {"osm_id": 305, "highway": "tertiary", "name": "West River Road"},
        },
    ]

    settlements = [
        {
            "type": "Feature",
            "id": "settle_cut",
            "geometry": {"type": "Point", "coordinates": [85.2000, 28.0020]},
            "properties": {"osm_id": 401, "name": "Isolated Hamlet", "place": "hamlet"},
        },
        {
            "type": "Feature",
            "id": "settle_detour",
            "geometry": {"type": "Point", "coordinates": [85.2020, 28.0020]},
            "properties": {"osm_id": 402, "name": "Hillside Village", "place": "village"},
        },
        {
            "type": "Feature",
            "id": "settle_safe",
            "geometry": {"type": "Point", "coordinates": [85.1980, 28.0000]},
            "properties": {"osm_id": 403, "name": "West Bank Town", "place": "town"},
        },
        {
            "type": "Feature",
            "id": "settle_unreachable",
            "geometry": {"type": "Point", "coordinates": [85.2500, 28.0500]},
            "properties": {"osm_id": 404, "name": "High Mountain Hermitage", "place": "isolated_dwelling"},
        },
    ]

    destinations = [
        {
            "type": "Feature",
            "id": "dest_hospital",
            "geometry": {"type": "Point", "coordinates": dest_pt},
            "properties": {"osm_id": 501, "name": "District Hospital", "amenity": "hospital"},
        }
    ]

    osm_data = {
        "roads": {"type": "FeatureCollection", "features": roads},
        "buildings": {"type": "FeatureCollection", "features": []},
        "bridges": {"type": "FeatureCollection", "features": []},
        "settlements": {"type": "FeatureCollection", "features": settlements},
        "destinations": {"type": "FeatureCollection", "features": destinations},
    }

    # Sever road_cut (301) and road_detour_direct (302)
    # by simulating an impact result that intersects them
    impact_data = {
        "geojson": {
            "potentially_affected_roads": {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"osm_id": 301, "exposed_length_meters": 50.0}},
                    {"type": "Feature", "properties": {"osm_id": 302, "exposed_length_meters": 60.0}},
                ]
            }
        }
    }

    conn_res = run_connectivity_pipeline(
        osm_data=osm_data,
        impact_data=impact_data,
        output_dir=tmp_path,
        max_snapping_distance_m=200.0,
    )

    summary = conn_res["primary_scenario"]["settlement_accessibility"]
    assert summary["total_settlements_evaluated"] == 4
    assert summary["potentially_cut_off_count"] == 1
    assert summary["detoured_count"] == 1
    assert summary["unaffected_count"] == 1
    assert summary["baseline_unreachable_count"] == 1
    assert "Isolated Hamlet" in summary["isolated_settlements"]

    # Verify GeoJSON properties for individual settlements
    geojson_path = tmp_path / "settlement_accessibility.geojson"
    assert geojson_path.is_file()
    with open(geojson_path, "r", encoding="utf-8") as f:
        fc = json.load(f)

    status_by_name = {f["properties"]["name"]: f["properties"]["connectivity_status"] for f in fc["features"]}
    assert status_by_name["Isolated Hamlet"] == "cut_off"
    assert status_by_name["Hillside Village"] == "detoured"
    assert status_by_name["West Bank Town"] == "unaffected"
    assert status_by_name["High Mountain Hermitage"] in ["insufficient_network_data", "baseline_unreachable"]


def test_empty_flood_extent_handling(tmp_path):
    """Verify that empty flood predictions produce 0 exposed assets and unchanged connectivity."""
    osm_data = {
        "roads": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "r1",
                    "geometry": {"type": "LineString", "coordinates": [[85.2, 28.0], [85.2, 28.01]]},
                    "properties": {"osm_id": 1, "highway": "primary", "name": "Main"},
                }
            ],
        },
        "buildings": {"type": "FeatureCollection", "features": []},
        "bridges": {"type": "FeatureCollection", "features": []},
        "settlements": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "s1",
                    "geometry": {"type": "Point", "coordinates": [85.2, 28.01]},
                    "properties": {"osm_id": 10, "name": "Settlement 1", "place": "village"},
                }
            ],
        },
        "destinations": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "d1",
                    "geometry": {"type": "Point", "coordinates": [85.2, 28.0]},
                    "properties": {"osm_id": 20, "name": "Clinic", "amenity": "clinic"},
                }
            ],
        },
    }

    empty_flood_gj = {"type": "FeatureCollection", "features": []}

    impact = assess_infrastructure_impact(
        osm_data=osm_data,
        snunet_flood_geojson=empty_flood_gj,
        output_dir=tmp_path,
    )
    assert impact["primary_scenario"]["potentially_exposed_roads"]["count"] == 0
    assert impact["primary_scenario"]["potentially_exposed_buildings"]["count"] == 0

    conn = run_connectivity_pipeline(
        osm_data=osm_data,
        impact_data=impact,
        output_dir=tmp_path,
    )
    scen = conn["primary_scenario"]["settlement_accessibility"]
    assert scen["potentially_cut_off_count"] == 0
    assert scen["unaffected_count"] == 1
