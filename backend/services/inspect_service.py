"""
Clickable Point Inspection Service
Samples genuine stored raster cells and spatial vectors for a given coordinate (lon, lat)
and produces calibrated physical measurements, model classifications,
infrastructure proximity context, and deterministic scientific explanations.
"""

import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import rasterio
from pyproj import Transformer
from shapely.geometry import Point, shape

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent


def inspect_point(
    run_id: str,
    lon: float,
    lat: float,
) -> Dict[str, Any]:
    """
    Samples all stored analysis rasters at (lon, lat) and returns
    calibrated measurements and deterministic explanation.
    """
    run_dir = WORKSPACE_ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Analysis run '{run_id}' not found.")

    manifest_path = run_dir / "manifest.json"
    manifest = {}
    if manifest_path.is_file():
        import json
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    pre_datetime = manifest.get("satellite_scenes", {}).get("pre_scene", {}).get("datetime", "2026-08-24T00:18:57Z")
    post_datetime = manifest.get("satellite_scenes", {}).get("post_scene", {}).get("datetime", "2026-09-05T00:18:57Z")

    rasters_dir = run_dir / "rasters"
    pre_vv_path = rasters_dir / "pre_vv.tif"
    if not pre_vv_path.is_file():
        raise FileNotFoundError(f"Rasters not found for run '{run_id}'.")

    # Determine projection and transform from pre_vv.tif
    with rasterio.open(pre_vv_path) as src:
        crs = src.crs
        transform = src.transform
        width = src.width
        height = src.height

        # Convert (lon, lat) WGS 84 -> metric CRS
        to_metric = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        x_m, y_m = to_metric.transform(lon, lat)

        # Get row, col
        r_raw, c_raw = rasterio.transform.rowcol(transform, x_m, y_m)
        row, col = int(r_raw), int(c_raw)

    is_inside = (0 <= row < height) and (0 <= col < width)
    if not is_inside:
        return {
            "coordinates": {"lon": lon, "lat": lat},
            "is_inside_aoi": False,
            "is_valid_data": False,
            "headline": "Outside Analyzed Footprint",
            "short_interpretation": "Selected coordinates lie outside the analyzed satellite imagery footprint.",
            "explanation": "Selected coordinates lie outside the analyzed satellite imagery footprint.",
            "disclaimer": "Analysis coverage is limited to the configured AOI bounds.",
        }

    # Sample raster pixel values
    def sample_val(tif_name: str, default: float = 0.0) -> float:
        p = rasters_dir / tif_name
        if not p.is_file():
            return default
        try:
            with rasterio.open(p) as rsrc:
                window = rasterio.windows.Window(col, row, 1, 1)
                data = rsrc.read(1, window=window)
                if data.size > 0:
                    val = float(data[0, 0])
                    return val if not np.isnan(val) else default
        except Exception:
            pass
        return default

    pre_vv_linear = sample_val("pre_vv.tif", 0.0)
    post_vv_linear = sample_val("post_vv.tif", 0.0)
    pre_vh_linear = sample_val("pre_vh.tif", 0.0)
    post_vh_linear = sample_val("post_vh.tif", 0.0)
    delta_vv_db = sample_val("delta_vv_db.tif", 0.0)
    snunet_flood_prob = sample_val("snunet_flood_prob.tif", 0.0)
    snunet_pred_class = int(round(sample_val("snunet_pred_class.tif", 0.0)))
    baseline_candidate = int(round(sample_val("baseline_candidate_flood.tif", 0.0)))
    dem_elevation_m = sample_val("dem.tif", 0.0)
    slope_deg = sample_val("slope.tif", 0.0)
    valid_flag = int(round(sample_val("valid_mask.tif", 1.0))) == 1

    pre_vv_db = round(10.0 * math.log10(max(pre_vv_linear, 1e-7)), 2) if pre_vv_linear > 0 else -30.0
    post_vv_db = round(10.0 * math.log10(max(post_vv_linear, 1e-7)), 2) if post_vv_linear > 0 else -30.0
    pre_vh_db = round(10.0 * math.log10(max(pre_vh_linear, 1e-7)), 2) if pre_vh_linear > 0 else -35.0
    post_vh_db = round(10.0 * math.log10(max(post_vh_linear, 1e-7)), 2) if post_vh_linear > 0 else -35.0

    class_names = {
        0: "No water (land/background)",
        1: "Permanent water",
        2: "Flood water",
    }
    class_label = class_names.get(snunet_pred_class, "Unknown")

    # Check nearby infrastructure context from Phase 3 vectors
    nearby_context = {}
    derived_dir = run_dir / "derived"
    osm_dir = run_dir / "osm"

    # Nearest settlement
    settle_path = derived_dir / "settlement_accessibility.geojson"
    if not settle_path.is_file():
        settle_path = osm_dir / "osm_settlements.geojson"

    if settle_path.is_file():
        try:
            import json
            with open(settle_path, "r", encoding="utf-8") as f:
                settle_fc = json.load(f)
            pt_wgs = Point(lon, lat)
            min_dist_m = float("inf")
            nearest_s = None

            for feat in settle_fc.get("features", []):
                s_geom = shape(feat["geometry"])
                # Approximate distance in meters at lat 28: 1 deg ~ 105,000m
                d_m = pt_wgs.distance(s_geom) * 105000.0
                if d_m < min_dist_m:
                    min_dist_m = d_m
                    nearest_s = feat

            if nearest_s and min_dist_m <= 8000.0:
                sprops = nearest_s["properties"]
                status_suffix = f" ({sprops.get('connectivity_status')})" if 'connectivity_status' in sprops else ""
                nearby_context["nearest_settlement"] = {
                    "name": sprops.get("name"),
                    "name_local": sprops.get("name_local", ""),
                    "status": sprops.get("connectivity_status", "unaffected"),
                    "distance_meters": int(round(min_dist_m)),
                    "description": f"{sprops.get('name')}{status_suffix}, {round(min_dist_m/1000.0, 1)} km away"
                }
        except Exception:
            pass

    # Nearest potentially affected road
    roads_path = derived_dir / "potentially_affected_roads.geojson"
    if roads_path.is_file():
        try:
            import json
            with open(roads_path, "r", encoding="utf-8") as f:
                roads_fc = json.load(f)
            pt_wgs = Point(lon, lat)
            min_road_dist_m = float("inf")
            nearest_r = None

            for feat in roads_fc.get("features", []):
                r_geom = shape(feat["geometry"])
                d_m = pt_wgs.distance(r_geom) * 105000.0
                if d_m < min_road_dist_m:
                    min_road_dist_m = d_m
                    nearest_r = feat

            if nearest_r and min_road_dist_m <= 2000.0:
                rprops = nearest_r["properties"]
                nearby_context["nearest_affected_road"] = {
                    "osm_id": rprops.get("osm_id"),
                    "name": rprops.get("name", "Unnamed Road"),
                    "highway": rprops.get("highway", "unknown"),
                    "exposed_length_m": rprops.get("exposed_length_meters", 0.0),
                    "distance_meters": int(round(min_road_dist_m)),
                }
        except Exception:
            pass

    # Deterministic scientific reasoning explanation
    explanation = ""
    headline = ""
    short_interpretation = ""
    disclaimer = "Absence of a predicted flood pixel does not confirm ground safety, vehicular passability, or structural integrity."

    if not valid_flag:
        headline = "Insufficient Valid Observations"
        short_interpretation = "Selected coordinates contain invalid or nodata values in one or more Sentinel-1 SAR passes."
        explanation = "Pixel is marked as invalid or nodata in one or more satellite observations."
    elif snunet_pred_class == 2:
        headline = "New Water-Consistent Evidence Detected"
        short_interpretation = f"SNUNet-ECAM deep Siamese model predicts standing flood water (confidence: {snunet_flood_prob*100.0:.1f}%) with a {abs(delta_vv_db):.1f} dB backscatter drop on {slope_deg:.1f}° slope."
        explanation = (
            f"Flood inundation predicted: Modelled backscatter drop of {delta_vv_db:.2f} dB "
            f"(pre: {pre_vv_db:.1f} dB, post: {post_vv_db:.1f} dB) on {slope_deg:.1f} deg terrain "
            f"(elevation {dem_elevation_m:.0f}m). SNUNet-ECAM deep Siamese model predicts flood water "
            f"with {snunet_flood_prob*100.0:.1f}% confidence."
        )
    elif snunet_pred_class == 1:
        headline = "Permanent Trishuli River Water"
        short_interpretation = f"Persistent low radar backscatter ({pre_vv_db:.1f} dB pre / {post_vv_db:.1f} dB post) mapped as permanent Trishuli river channel."
        explanation = (
            f"Permanent water detected: Persistent low backscatter in both pre-event ({pre_vv_db:.1f} dB) "
            f"and post-event ({post_vv_db:.1f} dB) radar passes. Mapped as permanent Trishuli river channel."
        )
    elif snunet_pred_class == 0:
        # Land / no water classified by deep Siamese model
        if delta_vv_db < -0.5:
            slope_excess = slope_deg >= 15.0
            if delta_vv_db >= -3.0 and slope_excess:
                headline = "No Significant Flood Water Detected"
                short_interpretation = f"Minor backscatter drop (-{abs(delta_vv_db):.1f} dB) on steep terrain ({slope_deg:.1f}°); SNUNet classifies as land/background."
                explanation = (
                    f"A backscatter decrease of {abs(delta_vv_db):.2f} dB is observed (pre: {pre_vv_db:.1f} dB, post: {post_vv_db:.1f} dB). "
                    f"The configured classical candidate criterion (threshold < -3.0 dB) is not met, and the terrain slope ({slope_deg:.1f} deg) "
                    f"exceeds the 15.0 deg slope filter. The SNUNet classification indicates no water (probability {snunet_flood_prob*100.0:.1f}%) at this point."
                )
            elif delta_vv_db < -3.0 and slope_excess:
                headline = "Radar Change Detected (Cause Uncertain: Topographic Shadow)"
                short_interpretation = f"Significant radar drop (-{abs(delta_vv_db):.1f} dB), but steep {slope_deg:.1f}° slope indicates radar geometric shadow rather than standing water; SNUNet classifies as land."
                explanation = (
                    f"Significant backscatter decrease of {abs(delta_vv_db):.2f} dB is observed (pre: {pre_vv_db:.1f} dB, post: {post_vv_db:.1f} dB). "
                    f"However, the terrain slope ({slope_deg:.1f} deg) exceeds the 15.0 deg slope filter (consistent with radar geometric shadow or mountain slope effects), "
                    f"and the SNUNet classification indicates no water (probability {snunet_flood_prob*100.0:.1f}%)."
                )
            elif delta_vv_db < -3.0 and not slope_excess:
                headline = "Radar Change Detected (Cause Uncertain: Soil Moisture / Agriculture)"
                short_interpretation = f"Classical radar drop (-{abs(delta_vv_db):.1f} dB) on gentle terrain ({slope_deg:.1f}°); SNUNet classifies as soil moisture or land wetting rather than standing flood water."
                explanation = (
                    f"Classical candidate change criterion is met (decrease of {abs(delta_vv_db):.2f} dB, slope {slope_deg:.1f} deg), "
                    f"but the SNUNet deep Siamese model classifies this point as no water (probability {snunet_flood_prob*100.0:.1f}%), "
                    f"consistent with soil moisture or agricultural change rather than standing flood water."
                )
            else:
                headline = "No Significant Change Detected"
                short_interpretation = f"Minor backscatter variation (-{abs(delta_vv_db):.1f} dB); normal background terrain under current analysis."
                explanation = (
                    f"A minor backscatter decrease of {abs(delta_vv_db):.2f} dB is observed (pre: {pre_vv_db:.1f} dB, post: {post_vv_db:.1f} dB). "
                    f"The configured classical candidate criterion (threshold < -3.0 dB) is not met, and the SNUNet classification "
                    f"indicates no water (probability {snunet_flood_prob*100.0:.1f}%)."
                )
        elif delta_vv_db >= 3.0:
            headline = "Radar Backscatter Increase Detected"
            short_interpretation = f"Radar backscatter increased by +{delta_vv_db:.1f} dB; consistent with surface roughening or debris deposit."
            explanation = (
                f"Radar backscatter increase of +{delta_vv_db:.2f} dB observed (pre: {pre_vv_db:.1f} dB, post: {post_vv_db:.1f} dB). "
                f"Consistent with surface roughening, debris deposit, or post-event moisture changes; SNUNet classifies this location as no water."
            )
        else:
            headline = "No Significant Change Detected"
            short_interpretation = f"SAR backscatter stable across observation window ({delta_vv_db:+.1f} dB); no flood inundation detected."
            explanation = (
                f"Minimal radar backscatter change observed across the observation baseline (pre: {pre_vv_db:.1f} dB, post: {post_vv_db:.1f} dB, "
                f"delta_VV: {delta_vv_db:+.2f} dB, slope: {slope_deg:.1f} deg). The SNUNet classification indicates no water (probability {snunet_flood_prob*100.0:.1f}%)."
            )

    return {
        "coordinates": {"lon": round(lon, 5), "lat": round(lat, 5)},
        "grid_pixel": {"row": row, "col": col},
        "is_inside_aoi": True,
        "is_valid_data": valid_flag,
        "headline": headline,
        "short_interpretation": short_interpretation,
        "disclaimer": disclaimer,
        "observation_dates": {
            "pre_datetime": pre_datetime,
            "post_datetime": post_datetime,
        },
        "measurements": {
            "pre_vv_db": pre_vv_db,
            "post_vv_db": post_vv_db,
            "pre_vh_db": pre_vh_db,
            "post_vh_db": post_vh_db,
            "delta_vv_db": round(delta_vv_db, 2),
            "snunet_flood_prob": round(snunet_flood_prob, 3),
            "snunet_class": snunet_pred_class,
            "snunet_class_name": class_label,
            "classical_candidate": bool(baseline_candidate == 1),
            "elevation_meters": round(dem_elevation_m, 1),
            "slope_degrees": round(slope_deg, 1),
        },
        "nearby_context": nearby_context,
        "explanation": explanation,
    }
