"""
Asset Impact Analysis Service
Calculates infrastructure exposure (roads, buildings, bridges) by intersecting
pre-event historical OSM assets with model-derived and classical flood extent polygons.
All spatial calculations are performed in projected metric coordinates (UTM EPSG:32645).
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from pyproj import Transformer
from shapely.geometry import shape, mapping, MultiPolygon, Polygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
import shapely


def reproject_geojson_feature(
    feature: Dict[str, Any],
    transformer: Transformer,
) -> Dict[str, Any]:
    """
    Transforms a GeoJSON feature's geometry using a PyProj transformer.
    """
    geom = shape(feature["geometry"])
    proj_geom = shapely.transform(geom, transformer.transform, interleaved=False)
    new_feat = dict(feature)
    new_feat["geometry"] = mapping(proj_geom)
    return new_feat


class ImpactAnalysisEngine:
    def __init__(self, metric_crs: str = "EPSG:32645"):
        self.metric_crs = metric_crs
        self.to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
        self.to_wgs84 = Transformer.from_crs(metric_crs, "EPSG:4326", always_xy=True)

    def _prepare_flood_union(self, flood_geojson: Dict[str, Any]) -> Tuple[Any, float]:
        """
        Reprojects flood polygons to metric UTM and computes a valid unary union.
        Returns (metric_union_geom, total_area_sqkm).
        """
        features = flood_geojson.get("features", [])
        if not features:
            return Polygon(), 0.0

        metric_polys = []
        for feat in features:
            geom = shape(feat["geometry"])
            if geom.is_empty:
                continue
            proj_geom = shapely.transform(geom, self.to_metric.transform, interleaved=False)
            if not proj_geom.is_valid:
                proj_geom = proj_geom.buffer(0)
            metric_polys.append(proj_geom)

        if not metric_polys:
            return Polygon(), 0.0

        union_geom = unary_union(metric_polys)
        total_area_sqkm = round(union_geom.area / 1e6, 4)
        return union_geom, total_area_sqkm

    def compute_exposure_scenario(
        self,
        osm_roads_geojson: Dict[str, Any],
        osm_buildings_geojson: Dict[str, Any],
        osm_bridges_geojson: Dict[str, Any],
        flood_geojson: Dict[str, Any],
        scenario_name: str = "snunet_primary",
        buffer_meters: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Computes exposure for roads, buildings, and bridges against a flood extent geometry.
        """
        flood_union_metric, flood_area_sqkm = self._prepare_flood_union(flood_geojson)
        if buffer_meters > 0.0 and not flood_union_metric.is_empty:
            eval_flood_metric = flood_union_metric.buffer(buffer_meters)
        else:
            eval_flood_metric = flood_union_metric

        exposed_roads = []
        total_exposed_road_length_m = 0.0
        total_road_length_m = 0.0

        # 1. Analyze Road Segments
        for road_feat in osm_roads_geojson.get("features", []):
            orig_geom = shape(road_feat["geometry"])
            if orig_geom.is_empty:
                continue

            proj_road = shapely.transform(orig_geom, self.to_metric.transform, interleaved=False)
            road_len_m = proj_road.length
            total_road_length_m += road_len_m

            if eval_flood_metric.is_empty:
                continue

            if proj_road.intersects(eval_flood_metric):
                intersection_geom = proj_road.intersection(eval_flood_metric)
                exposed_len_m = intersection_geom.length
                if exposed_len_m > 1.0:  # Ignore sub-meter numerical precision slivers
                    total_exposed_road_length_m += exposed_len_m
                    exposed_pct = round((exposed_len_m / road_len_m) * 100.0, 1) if road_len_m > 0 else 100.0

                    # Convert intersection back to WGS 84 GeoJSON for mapping
                    wgs_intersection = shapely.transform(intersection_geom, self.to_wgs84.transform, interleaved=False)
                    road_props = dict(road_feat.get("properties", {}))
                    road_props.update({
                        "exposure_status": "potentially_exposed",
                        "scenario": scenario_name,
                        "total_length_meters": round(road_len_m, 1),
                        "exposed_length_meters": round(exposed_len_m, 1),
                        "exposed_fraction_pct": exposed_pct,
                        "is_bridge": road_props.get("bridge", "no") in ["yes", "true", "suspension", "viaduct"],
                    })

                    exposed_roads.append({
                        "type": "Feature",
                        "id": f"exposed_road_{road_props.get('osm_id')}",
                        "geometry": mapping(orig_geom),  # Preserve full way geometry for routing visualization
                        "properties": road_props,
                    })

        # 2. Analyze Buildings
        exposed_buildings = []
        total_exposed_building_area_sqm = 0.0

        for bldg_feat in osm_buildings_geojson.get("features", []):
            orig_geom = shape(bldg_feat["geometry"])
            if orig_geom.is_empty:
                continue

            proj_bldg = shapely.transform(orig_geom, self.to_metric.transform, interleaved=False)
            if eval_flood_metric.is_empty:
                continue

            if proj_bldg.intersects(eval_flood_metric):
                inter_bldg = proj_bldg.intersection(eval_flood_metric)
                exp_area = inter_bldg.area
                total_exposed_building_area_sqm += exp_area

                bldg_props = dict(bldg_feat.get("properties", {}))
                bldg_props.update({
                    "exposure_status": "potentially_exposed",
                    "scenario": scenario_name,
                    "building_footprint_sqm": round(proj_bldg.area, 1),
                    "exposed_area_sqm": round(exp_area, 1),
                })
                exposed_buildings.append({
                    "type": "Feature",
                    "id": f"exposed_bldg_{bldg_props.get('osm_id')}",
                    "geometry": mapping(orig_geom),
                    "properties": bldg_props,
                })

        # 3. Analyze Bridges
        exposed_bridges = []
        for bridge_feat in osm_bridges_geojson.get("features", []):
            orig_geom = shape(bridge_feat["geometry"])
            if orig_geom.is_empty:
                continue

            proj_bridge = shapely.transform(orig_geom, self.to_metric.transform, interleaved=False)
            if eval_flood_metric.is_empty:
                continue

            if proj_bridge.intersects(eval_flood_metric):
                br_props = dict(bridge_feat.get("properties", {}))
                br_props.update({
                    "exposure_status": "potentially_exposed",
                    "scenario": scenario_name,
                    "bridge_length_meters": round(proj_bridge.length, 1),
                })
                exposed_bridges.append({
                    "type": "Feature",
                    "id": f"exposed_bridge_{br_props.get('osm_id')}",
                    "geometry": mapping(orig_geom),
                    "properties": br_props,
                })

        # Summary statistics
        summary = {
            "scenario": scenario_name,
            "flood_area_sqkm": flood_area_sqkm,
            "total_road_network_length_km": round(total_road_length_m / 1000.0, 2),
            "potentially_exposed_roads": {
                "count": len(exposed_roads),
                "unique_affected_osm_ways": len(exposed_roads),
                "total_exposed_length_meters": round(total_exposed_road_length_m, 1),
                "total_exposed_length_km": round(total_exposed_road_length_m / 1000.0, 3),
                "percentage_of_network": round((total_exposed_road_length_m / total_road_length_m) * 100.0, 2) if total_road_length_m > 0 else 0.0,
                "counting_unit": "unique OSM way features",
            },
            "potentially_exposed_buildings": {
                "count": len(exposed_buildings),
                "total_exposed_area_sqm": round(total_exposed_building_area_sqm, 1),
            },
            "potentially_exposed_bridges": {
                "count": len(exposed_bridges),
            },
            "scientific_caveat": (
                "Exposure indicates spatial intersection with modeled inundation extent. "
                "Classified strictly as 'potentially_exposed' and does not confirm structural destruction."
            )
        }

        return {
            "summary": summary,
            "exposed_roads_geojson": {"type": "FeatureCollection", "features": exposed_roads},
            "exposed_buildings_geojson": {"type": "FeatureCollection", "features": exposed_buildings},
            "exposed_bridges_geojson": {"type": "FeatureCollection", "features": exposed_bridges},
        }


def assess_infrastructure_impact(
    osm_data: Dict[str, Any],
    snunet_flood_geojson: Dict[str, Any],
    baseline_change_geojson: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Executes infrastructure impact analysis for both the primary SNUNet scenario
    and the optional classical candidate radar change sensitivity scenario.
    Saves output GeoJSONs and summary metadata under output_dir.
    """
    engine = ImpactAnalysisEngine()
    geojson_dict = osm_data.get("geojson", osm_data)
    roads_gj = geojson_dict.get("roads", {"type": "FeatureCollection", "features": []})
    buildings_gj = geojson_dict.get("buildings", {"type": "FeatureCollection", "features": []})
    bridges_gj = geojson_dict.get("bridges", {"type": "FeatureCollection", "features": []})

    # 1. Primary Scenario: SNUNet Neural Network Flood Extent
    primary_res = engine.compute_exposure_scenario(
        osm_roads_geojson=roads_gj,
        osm_buildings_geojson=buildings_gj,
        osm_bridges_geojson=bridges_gj,
        flood_geojson=snunet_flood_geojson,
        scenario_name="snunet_primary",
    )

    # 2. Sensitivity Scenario: Classical Radar Backscatter Drop Baseline (if provided)
    sensitivity_res = None
    if baseline_change_geojson is not None:
        sensitivity_res = engine.compute_exposure_scenario(
            osm_roads_geojson=roads_gj,
            osm_buildings_geojson=buildings_gj,
            osm_bridges_geojson=bridges_gj,
            flood_geojson=baseline_change_geojson,
            scenario_name="change_sensitivity",
        )

    # Combine exposed assets for unified mapping layer
    combined_exposed_features = []
    combined_exposed_features.extend(primary_res["exposed_roads_geojson"]["features"])
    combined_exposed_features.extend(primary_res["exposed_buildings_geojson"]["features"])
    combined_exposed_features.extend(primary_res["exposed_bridges_geojson"]["features"])

    combined_impact_geojson = {
        "type": "FeatureCollection",
        "properties": {
            "scenario": "snunet_primary",
            "exposed_roads_count": primary_res["summary"]["potentially_exposed_roads"]["count"],
            "exposed_buildings_count": primary_res["summary"]["potentially_exposed_buildings"]["count"],
            "exposed_bridges_count": primary_res["summary"]["potentially_exposed_bridges"]["count"],
        },
        "features": combined_exposed_features,
    }

    result = {
        "status": "COMPLETED",
        "primary_scenario": primary_res["summary"],
        "sensitivity_scenario": sensitivity_res["summary"] if sensitivity_res else None,
        "geojson": {
            "asset_impact": combined_impact_geojson,
            "potentially_affected_roads": primary_res["exposed_roads_geojson"],
            "potentially_affected_buildings": primary_res["exposed_buildings_geojson"],
            "potentially_affected_bridges": primary_res["exposed_bridges_geojson"],
        }
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        impact_path = output_dir / "asset_impact.geojson"
        roads_path = output_dir / "potentially_affected_roads.geojson"

        with open(impact_path, "w", encoding="utf-8") as f:
            json.dump(combined_impact_geojson, f, indent=2)

        with open(roads_path, "w", encoding="utf-8") as f:
            json.dump(primary_res["exposed_roads_geojson"], f, indent=2)

        result["file_paths"] = {
            "asset_impact_geojson": str(impact_path),
            "potentially_affected_roads_geojson": str(roads_path),
        }

    return result
