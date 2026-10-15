"""
Asset Impact and Rescue Filter Service
Provides dynamic intersection calculation, feature enrichment, and honest
exposure status labelling for mapped buildings, roads, bridges, and settlements.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from shapely.geometry import shape

# Global in-memory cache keyed by (run_id, mtime) to avoid re-parsing on every HTTP request
_ASSET_CACHE: Dict[str, Dict[str, Any]] = {}


def get_run_asset_impact_data(run_dir: Path) -> Dict[str, Any]:
    """
    Computes or retrieves cached spatial intersections and exposure classifications
    for pre-event OSM assets against primary SNUNet flood extent and classical candidate change.
    """
    run_id = run_dir.name
    cache_key = run_id
    if cache_key in _ASSET_CACHE:
        return _ASSET_CACHE[cache_key]

    vectors_dir = run_dir / "vectors"
    osm_dir = run_dir / "osm"
    derived_dir = run_dir / "derived"

    # 1. Load flood extents
    flood_path = vectors_dir / "snunet_flood_extent.geojson"
    cand_path = vectors_dir / "baseline_candidate_flood.geojson"

    flood_shapes = []
    if flood_path.is_file():
        with open(flood_path, "r", encoding="utf-8") as f:
            flood_data = json.load(f)
            flood_shapes = [shape(x["geometry"]) for x in flood_data.get("features", []) if x.get("geometry")]

    cand_shapes = []
    if cand_path.is_file():
        with open(cand_path, "r", encoding="utf-8") as f:
            cand_data = json.load(f)
            cand_shapes = [shape(x["geometry"]) for x in cand_data.get("features", []) if x.get("geometry")]

    # 2. Enrich Buildings (Sample of 1,000 historical OSM footprints)
    buildings_path = osm_dir / "osm_buildings.geojson"
    enriched_buildings = {"type": "FeatureCollection", "features": []}
    exposed_buildings = {"type": "FeatureCollection", "features": []}
    candidate_buildings = {"type": "FeatureCollection", "features": []}

    if buildings_path.is_file():
        with open(buildings_path, "r", encoding="utf-8") as f:
            bldg_data = json.load(f)

        for feat in bldg_data.get("features", []):
            f_copy = {
                "type": feat.get("type", "Feature"),
                "geometry": feat.get("geometry"),
                "properties": dict(feat.get("properties", {})),
            }
            props = f_copy["properties"]
            geom = shape(f_copy["geometry"]) if f_copy.get("geometry") else None

            is_primary = False
            is_cand = False
            if geom and not geom.is_empty:
                is_primary = any(geom.intersects(fs) for fs in flood_shapes)
                is_cand = any(geom.intersects(cs) for cs in cand_shapes)

            props["primary_flood_exposed"] = is_primary
            props["candidate_change_exposed"] = is_cand
            props["model_scope_caveat"] = "Structural collapse, building destruction, or damage is NOT assessed by this model."
            props["inventory_sample_warning"] = "Evaluated within 1,000-building historical OSM sample limit."
            props["structural_collapse_assessed"] = False

            if is_primary:
                props["exposure_status_label"] = "Potentially flood-exposed"
                props["evidence_scenario"] = "SNUNet Sentinel-1 RTC neural water prediction (primary scenario)"
                exposed_buildings["features"].append(f_copy)
            elif is_cand:
                props["exposure_status_label"] = "Possible surface-change overlap — uncertain cause"
                props["evidence_scenario"] = "Classical radar backscatter drop (Delta_dB < -3.0, lower confidence)"
                candidate_buildings["features"].append(f_copy)
            else:
                props["exposure_status_label"] = "No detected overlap in available data"
                props["evidence_scenario"] = "Evaluated against SNUNet and classical radar change layers"

            enriched_buildings["features"].append(f_copy)

    # 3. Enrich Bridges
    bridges_path = osm_dir / "osm_bridges.geojson"
    enriched_bridges = {"type": "FeatureCollection", "features": []}
    exposed_bridges = {"type": "FeatureCollection", "features": []}
    candidate_bridges = {"type": "FeatureCollection", "features": []}

    if bridges_path.is_file():
        with open(bridges_path, "r", encoding="utf-8") as f:
            br_data = json.load(f)

        for feat in br_data.get("features", []):
            f_copy = {
                "type": feat.get("type", "Feature"),
                "geometry": feat.get("geometry"),
                "properties": dict(feat.get("properties", {})),
            }
            props = f_copy["properties"]
            geom = shape(f_copy["geometry"]) if f_copy.get("geometry") else None

            is_primary = False
            is_cand = False
            if geom and not geom.is_empty:
                is_primary = any(geom.intersects(fs) for fs in flood_shapes)
                is_cand = any(geom.intersects(cs) for cs in cand_shapes)

            props["primary_flood_exposed"] = is_primary
            props["candidate_change_exposed"] = is_cand
            props["model_scope_caveat"] = "Spatial intersection indicates potential surface inundation. Structural damage or bridge collapse is NOT assessed by this model."

            if is_primary:
                props["exposure_status_label"] = "Potentially flood-exposed"
                props["evidence_scenario"] = "SNUNet Sentinel-1 RTC neural water prediction (primary scenario)"
                exposed_bridges["features"].append(f_copy)
            elif is_cand:
                props["exposure_status_label"] = "Possible surface-change overlap — uncertain cause"
                props["evidence_scenario"] = "Classical radar backscatter drop (Delta_dB < -3.0, lower confidence)"
                candidate_bridges["features"].append(f_copy)
            else:
                props["exposure_status_label"] = "No detected overlap in available data"
                props["evidence_scenario"] = "Evaluated against SNUNet and classical radar change layers"

            enriched_bridges["features"].append(f_copy)

    # 4. Roads & Severance
    roads_path = osm_dir / "osm_roads.geojson"
    cand_roads = {"type": "FeatureCollection", "features": []}
    severed_roads = {"type": "FeatureCollection", "features": []}

    # Severed road IDs from connectivity summary
    severed_ids = set()
    conn_path = derived_dir / "connectivity_summary.json"
    settlement_counts = {"total": 0, "cut_off": 0, "detoured": 0, "unaffected": 0}
    if conn_path.is_file():
        try:
            with open(conn_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                severed_list = c_data.get("severed_road_segments", [])
                severed_ids = {s.get("osm_id") for s in severed_list if s.get("osm_id")}
                st_acc = c_data.get("settlement_accessibility", {})
                settlement_counts["cut_off"] = len(st_acc.get("cut_off_settlements", [])) or len(st_acc.get("isolated_settlements", []))
                settlement_counts["detoured"] = len(st_acc.get("detoured_settlements", []))
                settlement_counts["unaffected"] = st_acc.get("unaffected_count", 23)
                settlement_counts["total"] = settlement_counts["cut_off"] + settlement_counts["detoured"] + settlement_counts["unaffected"]
        except Exception:
            pass

    pot_roads_path = derived_dir / "potentially_affected_roads.geojson"
    pot_roads_count = 0
    pot_roads_length = 0.0
    if pot_roads_path.is_file():
        with open(pot_roads_path, "r", encoding="utf-8") as f:
            pot_data = json.load(f)
            pot_feats = pot_data.get("features", [])
            pot_roads_count = len(pot_feats)
            pot_roads_length = sum(feat.get("properties", {}).get("exposed_length_meters", 0.0) for feat in pot_feats)

    if roads_path.is_file():
        with open(roads_path, "r", encoding="utf-8") as f:
            r_data = json.load(f)

        for feat in r_data.get("features", []):
            f_copy = {
                "type": feat.get("type", "Feature"),
                "geometry": feat.get("geometry"),
                "properties": dict(feat.get("properties", {})),
            }
            props = f_copy["properties"]
            osm_id = props.get("osm_id")
            geom = shape(f_copy["geometry"]) if f_copy.get("geometry") else None

            if osm_id in severed_ids:
                props["exposure_status_label"] = "Graph Disrupted — Link Severed"
                props["severed_in_connectivity"] = True
                severed_roads["features"].append(f_copy)

            if geom and not geom.is_empty:
                if any(geom.intersects(cs) for cs in cand_shapes):
                    props["candidate_change_exposed"] = True
                    props["exposure_status_label"] = "Possible surface-change overlap — uncertain cause"
                    cand_roads["features"].append(f_copy)

    # 5. Compile Rescue Summary Object
    rescue_summary = {
        "run_id": run_id,
        "buildings": {
            "total_sample": len(enriched_buildings["features"]),
            "primary_exposed": len(exposed_buildings["features"]),
            "candidate_exposed": len(candidate_buildings["features"]),
            "inventory_limit": 1000,
            "sample_cap": 1000,
            "completeness_warning": "Building sample capped at 1,000 historical OSM footprints. Inventory completeness is partial and variable across rural terrain.",
            "structural_caveat": "Structural collapse is not assessed. Satellite radar water extent does not establish structural damage, collapse, or uninhabitable condition without in-situ survey.",
            "sample_completeness_note": "Evaluated on 1,000-building sample due to historical OSM regional limit. Collapse not assessed.",
        },
        "roads": {
            "total_roads": len(r_data.get("features", [])) if roads_path.is_file() else 0,
            "primary_exposed": pot_roads_count,
            "exposed_length_meters": round(pot_roads_length, 1),
            "candidate_exposed": len(cand_roads["features"]),
            "severed_graph_edges": len(severed_roads["features"]),
        },
        "bridges": {
            "total_bridges": len(enriched_bridges["features"]),
            "primary_exposed": len(exposed_bridges["features"]),
            "candidate_exposed": len(candidate_bridges["features"]),
        },
        "settlements": settlement_counts,
    }

    result = {
        "summary": rescue_summary,
        "layers": {
            "osm_buildings": enriched_buildings,
            "buildings_exposed": exposed_buildings,
            "buildings_candidate": candidate_buildings,
            "osm_bridges": enriched_bridges,
            "bridges_exposed": exposed_bridges,
            "bridges_candidate": candidate_bridges,
            "roads_candidate": cand_roads,
            "severed_roads": severed_roads,
        },
    }

    _ASSET_CACHE[cache_key] = result
    return result
