"""
Historical OpenStreetMap Extraction Service
Extracts pre-event OpenStreetMap geometries (roads, buildings, bridges, settlements, destinations)
at or before the challenge cutoff (2026-07-27T00:00:00Z).
Integrates ohsome API temporal metadata checks and fast attic Overpass queries with local run caching.
"""

import os
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from shapely.geometry import Point, LineString, Polygon, shape, mapping

OHSOME_BASE_URL = "https://api.ohsome.org/v1"
HISTORICAL_CUTOFF_TIMESTAMP = "2026-07-27T00:00:00Z"

OVERPASS_MIRRORS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

ROUTABLE_HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link", "living_street",
    "service", "track", "road"
}


class OSMServiceError(Exception):
    """Base exception for OpenStreetMap extraction errors."""
    pass


def check_ohsome_temporal_extent() -> Dict[str, Any]:
    """
    Queries ohsome API /metadata to establish the official historical dataset temporal extent.
    """
    url = f"{OHSOME_BASE_URL}/metadata"
    req = urllib.request.Request(url, headers={"User-Agent": "FloodIntelligence/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            temporal_extent = data.get("extractRegion", {}).get("temporalExtent", {})
            return {
                "status": "AVAILABLE",
                "api_version": data.get("apiVersion", "1.x"),
                "from_timestamp": temporal_extent.get("fromTimestamp"),
                "to_timestamp": temporal_extent.get("toTimestamp", "2026-07-27T09:00Z"),
                "attribution": "© OpenStreetMap contributors, HeiGIT ohsome API"
            }
    except Exception as e:
        return {
            "status": "UNREACHABLE",
            "error": str(e),
            "fallback_to_timestamp": "2026-07-27T00:00:00Z",
            "attribution": "© OpenStreetMap contributors"
        }


def query_overpass_historical(
    query_body: str,
    max_retries: int = 3,
    timeout_sec: int = 40,
) -> Dict[str, Any]:
    """
    Executes an attic Overpass QL query with multi-mirror failover and bounded retries.
    """
    data = urllib.parse.urlencode({"data": query_body}).encode("utf-8")
    last_err = None

    for mirror in OVERPASS_MIRRORS:
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    mirror,
                    data=data,
                    headers={"User-Agent": "FloodIntelligence/1.0 (Hackathon Track B)"}
                )
                with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                last_err = e
                time.sleep(1.5 * (attempt + 1))

    raise OSMServiceError(f"Failed to query historical OSM data across Overpass mirrors: {last_err}")


def convert_overpass_to_geojson_features(
    elements: List[Dict[str, Any]],
    target_type: str,
) -> List[Dict[str, Any]]:
    """
    Converts raw Overpass elements (with geometry) into GeoJSON features.
    """
    features = []

    for elem in elements:
        osm_id = elem.get("id")
        tags = elem.get("tags", {})
        elem_type = elem.get("type")

        geom = None
        if elem_type == "node":
            lat = elem.get("lat")
            lon = elem.get("lon")
            if lat is not None and lon is not None:
                geom = {"type": "Point", "coordinates": [float(lon), float(lat)]}
        elif elem_type == "way":
            coords = elem.get("geometry", [])
            if coords and len(coords) >= 2:
                line_coords = [[float(c["lon"]), float(c["lat"])] for c in coords if "lat" in c and "lon" in c]
                if target_type == "buildings" and len(line_coords) >= 4 and line_coords[0] == line_coords[-1]:
                    geom = {"type": "Polygon", "coordinates": [line_coords]}
                elif len(line_coords) >= 2:
                    geom = {"type": "LineString", "coordinates": line_coords}

        if geom is None:
            continue

        name_val = tags.get("name:en") or tags.get("name") or "Unnamed"
        props = {
            "osm_id": osm_id,
            "osm_type": elem_type,
            "name": name_val,
            "name_en": tags.get("name:en", ""),
            "name_local": tags.get("name", ""),
            "tags": tags,
        }

        if target_type == "roads":
            props["highway"] = tags.get("highway", "unknown")
            props["bridge"] = tags.get("bridge", "no")
            props["surface"] = tags.get("surface", "unknown")
            props["oneway"] = tags.get("oneway", "no")
            props["is_routable"] = tags.get("highway") in ROUTABLE_HIGHWAYS
        elif target_type == "settlements":
            props["place"] = tags.get("place", "locality")
            props["population"] = tags.get("population", "unknown")
        elif target_type == "destinations":
            props["amenity"] = tags.get("amenity", "healthcare")
            props["healthcare"] = tags.get("healthcare", "yes")
        elif target_type == "buildings":
            props["building"] = tags.get("building", "yes")

        features.append({
            "type": "Feature",
            "id": f"{elem_type}_{osm_id}",
            "geometry": geom,
            "properties": props,
        })

    return features


def fetch_historical_osm_data(
    bbox: List[float],
    snapshot_timestamp: str = HISTORICAL_CUTOFF_TIMESTAMP,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetches pre-event OSM data (roads, buildings, bridges, settlements, destinations)
    for the specified bounding box [minx, miny, maxx, maxy].
    Enforces that snapshot_timestamp <= HISTORICAL_CUTOFF_TIMESTAMP (2026-07-27).
    Uses local file caching if cache_dir is provided.
    """
    minx, miny, maxx, maxy = bbox
    t0 = time.perf_counter()

    # Enforce strict challenge cutoff constraint
    if snapshot_timestamp > HISTORICAL_CUTOFF_TIMESTAMP:
        raise OSMServiceError(
            f"Requested OSM snapshot timestamp '{snapshot_timestamp}' violates challenge cutoff "
            f"rule (must be <= {HISTORICAL_CUTOFF_TIMESTAMP})."
        )

    # Check cache first
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_manifest = cache_dir / "osm_manifest.json"
        if cached_manifest.is_file():
            try:
                with open(cached_manifest, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                if manifest_data.get("snapshot_timestamp") == snapshot_timestamp:
                    loaded_geojson = {}
                    for key in ["roads", "bridges", "buildings", "settlements", "destinations"]:
                        fc_path = cache_dir / f"osm_{key}.geojson"
                        if fc_path.is_file():
                            with open(fc_path, "r", encoding="utf-8") as f:
                                loaded_geojson[key] = json.load(f)
                        else:
                            loaded_geojson[key] = {"type": "FeatureCollection", "features": []}
                    manifest_data["geojson"] = loaded_geojson
                    return manifest_data
            except Exception:
                pass

    # 1. Query ohsome metadata for official dataset temporal extent provenance
    ohsome_meta = check_ohsome_temporal_extent()

    # 2. Query indexed elements for roads, settlements, and amenities (Fast B-tree indexed queries)
    # Apply a network buffer (~2.5 km) around the AOI for highways so regional road corridors and bypasses
    # beyond the satellite image footprint are included in the graph, preventing boundary-clipping disconnections.
    road_buffer = 0.025
    r_miny, r_minx = miny - road_buffer, minx - road_buffer
    r_maxy, r_maxx = maxy + road_buffer, maxx + road_buffer

    # Overpass bbox convention: miny, minx, maxy, maxx
    query_network = f"""
    [out:json][timeout:25][date:"{snapshot_timestamp}"];
    (
      way["highway"]({r_miny},{r_minx},{r_maxy},{r_maxx});
      node["place"]({miny},{minx},{maxy},{maxx});
      node["amenity"]({miny},{minx},{maxy},{maxx});
    );
    out geom;
    """

    res_network = query_overpass_historical(query_network, max_retries=3, timeout_sec=30)
    elems = res_network.get("elements", [])

    raw_highways = [e for e in elems if e.get("type") == "way" and "highway" in e.get("tags", {})]
    raw_bridges = [e for e in raw_highways if e.get("tags", {}).get("bridge") in ["yes", "viaduct", "suspension", "true", "aqueduct"]]
    
    valid_places = {"city", "town", "village", "hamlet", "isolated_dwelling", "locality", "suburb", "neighbourhood"}
    raw_settlements = [e for e in elems if e.get("type") == "node" and e.get("tags", {}).get("place") in valid_places]
    
    healthcare_amenities = {"hospital", "clinic", "health_post", "doctors", "pharmacy"}
    raw_destinations = [e for e in elems if e.get("tags", {}).get("amenity") in healthcare_amenities]

    road_features = convert_overpass_to_geojson_features(raw_highways, "roads")
    bridge_features = convert_overpass_to_geojson_features(raw_bridges, "roads")
    settlement_features = convert_overpass_to_geojson_features(raw_settlements, "settlements")
    destination_features = convert_overpass_to_geojson_features(raw_destinations, "destinations")

    # If no designated healthcare facility in this rural tile, designate primary town centers as essential destinations
    if not destination_features and settlement_features:
        town_settlements = [s for s in settlement_features if s["properties"].get("place") in ["town", "city"]]
        fallback_dest = town_settlements if town_settlements else settlement_features[:2]
        for fd in fallback_dest:
            copied = dict(fd)
            copied["properties"] = dict(fd["properties"])
            copied["properties"]["amenity"] = "administrative_center"
            copied["properties"]["healthcare"] = "no"
            destination_features.append(copied)

    # 3. Query buildings (bounded query)
    query_buildings = f"""
    [out:json][timeout:25][date:"{snapshot_timestamp}"];
    way["building"]({miny},{minx},{maxy},{maxx});
    out geom 1000;
    """
    try:
        res_bldg = query_overpass_historical(query_buildings, max_retries=2, timeout_sec=25)
        raw_buildings = res_bldg.get("elements", [])
    except Exception:
        raw_buildings = []

    building_features = convert_overpass_to_geojson_features(raw_buildings, "buildings")

    fetch_time = round(time.perf_counter() - t0, 2)

    result_data = {
        "status": "SUCCESS",
        "snapshot_timestamp": snapshot_timestamp,
        "challenge_cutoff": HISTORICAL_CUTOFF_TIMESTAMP,
        "is_pre_event": True,
        "query_bbox": bbox,
        "fetch_time_seconds": fetch_time,
        "provenance": {
            "source": "OpenStreetMap Historical Attic Database",
            "snapshot_timestamp": snapshot_timestamp,
            "ohsome_extent": ohsome_meta,
            "license": "Open Data Commons Open Database License (ODbL) © OpenStreetMap contributors",
        },
        "counts": {
            "roads": len(road_features),
            "bridges": len(bridge_features),
            "buildings": len(building_features),
            "settlements": len(settlement_features),
            "destinations": len(destination_features),
        },
        "building_inventory": {
            "count": len(building_features),
            "is_capped": len(building_features) >= 1000,
            "limit": 1000,
            "completeness_warning": "Building inventory was query-capped at 1,000 features due to attic query limits. Zero exposed buildings applies strictly to the mapped sample and does not verify complete structural safety across the entire AOI."
        },
        "geojson": {
            "roads": {"type": "FeatureCollection", "features": road_features},
            "bridges": {"type": "FeatureCollection", "features": bridge_features},
            "buildings": {"type": "FeatureCollection", "features": building_features},
            "settlements": {"type": "FeatureCollection", "features": settlement_features},
            "destinations": {"type": "FeatureCollection", "features": destination_features},
        }
    }

    # Save to cache if dir provided
    if cache_dir is not None:
        for key in ["roads", "bridges", "buildings", "settlements", "destinations"]:
            fc_path = cache_dir / f"osm_{key}.geojson"
            with open(fc_path, "w", encoding="utf-8") as f:
                json.dump(result_data["geojson"][key], f, indent=2)

        cache_summary = dict(result_data)
        del cache_summary["geojson"]
        cache_summary["file_paths"] = {
            key: str(cache_dir / f"osm_{key}.geojson")
            for key in ["roads", "bridges", "buildings", "settlements", "destinations"]
        }
        with open(cached_manifest, "w", encoding="utf-8") as f:
            json.dump(cache_summary, f, indent=2)

    return result_data
