"""
Core Geospatial and SNUNet Inference Pipeline Orchestrator
Executes end-to-end satellite acquisition, windowed raster extraction, DEM/slope alignment,
tiled neural inference, classical change detection, vector polygonization, and manifest generation.
"""

import os
import sys
import time
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import rasterio

from backend.models.snunet import (
    CLASS_NO_WATER,
    CLASS_PERMANENT_WATER,
    CLASS_FLOOD_WATER,
    CLASS_NAMES,
    IMG_MEANS,
    IMG_STDS,
    DEM_MEAN,
    DEM_STD,
    SLOPE_MEAN,
    SLOPE_STD,
)
from backend.services.stac_service import (
    select_best_scene_pair,
    get_signed_copernicus_dem_urls,
    aoi_to_bbox_and_geom,
    NoCompatibleScenePairError,
)
from backend.services.raster_service import (
    extract_and_align_rasters,
    save_geotiff,
)
from backend.services.inference_service import InferenceEngine
from backend.services.baseline_service import (
    compute_radar_change_baseline,
    BaselineConfig,
)
from backend.services.polygon_service import (
    mask_to_geojson_features,
    save_geojson,
)
from backend.services.osm_service import (
    fetch_historical_osm_data,
    HISTORICAL_CUTOFF_TIMESTAMP,
    OSMServiceError,
)
from backend.services.impact_service import assess_infrastructure_impact
from backend.services.connectivity_service import run_connectivity_pipeline

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def run_flood_analysis(
    aoi: Union[List[float], Dict[str, Any]],
    event_date: str = "2026-08-26",
    output_base_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    search_window_days: int = 25,
    min_aoi_coverage_pct: float = 75.0,
    preferred_orbit: Optional[int] = None,
    preferred_orbit_state: Optional[str] = None,
    db_drop_threshold: float = -3.0,
    max_slope_degrees: float = 15.0,
    patch_size: int = 224,
    stride: int = 168,
    batch_size: int = 4,
    target_res: float = 10.0,
    run_infrastructure: bool = False,
) -> Dict[str, Any]:
    """
    Executes the complete Phase 2 analytical pipeline.
    """
    t_start = time.perf_counter()
    bbox, aoi_geom = aoi_to_bbox_and_geom(aoi)

    # 1. Setup run directory under runs/
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"flood_{run_timestamp}_{uuid.uuid4().hex[:6]}"

    if output_base_dir is None:
        output_dir = WORKSPACE_ROOT / "runs" / run_id
    else:
        output_dir = Path(output_base_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        checkpoint_path = WORKSPACE_ROOT / "checkpoints" / "best_segmentation.pt"

    timings = {}

    # 2. Step 1: STAC Query and Scene Pairing
    t0 = time.perf_counter()
    pair_meta = select_best_scene_pair(
        aoi=bbox,
        event_date_str=event_date,
        search_window_days=search_window_days,
        min_aoi_coverage_pct=min_aoi_coverage_pct,
        preferred_orbit=preferred_orbit,
        preferred_orbit_state=preferred_orbit_state,
    )
    dem_meta_list = get_signed_copernicus_dem_urls(bbox)
    dem_signed_urls = [d["signed_href"] for d in dem_meta_list]
    timings["stac_query_seconds"] = round(time.perf_counter() - t0, 2)

    # 3. Step 2: Windowed Raster Extraction & Alignment
    t0 = time.perf_counter()
    raster_dir = output_dir / "rasters"
    raster_res = extract_and_align_rasters(
        pre_signed_urls=pair_meta["pre_signed_urls"],
        post_signed_urls=pair_meta["post_signed_urls"],
        dem_signed_urls=dem_signed_urls,
        aoi_bbox=bbox,
        output_dir=raster_dir,
        target_res=target_res,
    )
    timings["raster_extraction_seconds"] = round(time.perf_counter() - t0, 2)

    arrays = raster_res["arrays"]
    master_crs = rasterio.crs.CRS.from_string(raster_res["crs"])
    master_transform = rasterio.Affine(*raster_res["transform"])
    valid_mask = arrays["valid_mask"]

    # 4. Step 3: Tiled SNUNet Inference
    t0 = time.perf_counter()
    engine = InferenceEngine(
        checkpoint_path=checkpoint_path,
        patch_size=patch_size,
        stride=stride,
        batch_size=batch_size,
    )
    pre_3ch, post_3ch = engine.prepare_input_tensors(
        pre_vv=arrays["pre_vv"],
        pre_vh=arrays["pre_vh"],
        post_vv=arrays["post_vv"],
        post_vh=arrays["post_vh"],
        dem=arrays["dem"],
    )
    inf_res = engine.predict_tiled(pre_3ch, post_3ch, valid_mask=valid_mask)
    timings["neural_inference_seconds"] = round(time.perf_counter() - t0, 2)

    # Save model rasters
    prob_path = save_geotiff(
        raster_dir / "snunet_flood_prob.tif",
        inf_res["flood_prob"],
        master_transform,
        master_crs,
        nodata=0.0,
        dtype="float32",
    )
    class_path = save_geotiff(
        raster_dir / "snunet_pred_class.tif",
        inf_res["pred_class"],
        master_transform,
        master_crs,
        nodata=0,
        dtype="uint8",
    )

    # 5. Step 4: Independent Classical Radar Change Baseline
    t0 = time.perf_counter()
    baseline_cfg = BaselineConfig(
        db_drop_threshold=db_drop_threshold,
        max_slope_degrees=max_slope_degrees,
    )
    base_res = compute_radar_change_baseline(
        pre_vv_linear=arrays["pre_vv"],
        post_vv_linear=arrays["post_vv"],
        slope_deg=arrays["slope"],
        valid_mask=valid_mask,
        config=baseline_cfg,
    )
    timings["classical_baseline_seconds"] = round(time.perf_counter() - t0, 2)

    # Save baseline rasters
    delta_db_path = save_geotiff(
        raster_dir / "delta_vv_db.tif",
        base_res["delta_vv_db"],
        master_transform,
        master_crs,
        nodata=0.0,
        dtype="float32",
    )
    baseline_candidate_path = save_geotiff(
        raster_dir / "baseline_candidate_flood.tif",
        base_res["candidate_flood_mask"],
        master_transform,
        master_crs,
        nodata=0,
        dtype="uint8",
    )

    # 6. Step 5: Vector Polygonization & Area Statistics
    t0 = time.perf_counter()
    vector_dir = output_dir / "vectors"

    # SNUNet Flood Extent (Class 2)
    snunet_flood_bin = (inf_res["pred_class"] == CLASS_FLOOD_WATER).astype(np.uint8)
    snunet_geojson = mask_to_geojson_features(
        snunet_flood_bin,
        master_transform,
        master_crs,
        target_value=1,
        min_area_sqm=500.0,
    )
    snunet_geojson_path = save_geojson(snunet_geojson, vector_dir / "snunet_flood_extent.geojson")

    # SNUNet Permanent Water Extent (Class 1)
    snunet_perm_bin = (inf_res["pred_class"] == CLASS_PERMANENT_WATER).astype(np.uint8)
    snunet_perm_geojson = mask_to_geojson_features(
        snunet_perm_bin,
        master_transform,
        master_crs,
        target_value=1,
        min_area_sqm=500.0,
    )
    snunet_perm_geojson_path = save_geojson(snunet_perm_geojson, vector_dir / "snunet_permanent_water.geojson")

    # Classical Baseline Candidate Extent
    baseline_geojson = mask_to_geojson_features(
        base_res["candidate_flood_mask"],
        master_transform,
        master_crs,
        target_value=1,
        min_area_sqm=500.0,
    )
    baseline_geojson_path = save_geojson(baseline_geojson, vector_dir / "baseline_candidate_flood.geojson")
    timings["vector_polygonization_seconds"] = round(time.perf_counter() - t0, 2)

    total_duration_sec = round(time.perf_counter() - t_start, 2)
    timings["total_pipeline_seconds"] = total_duration_sec

    # Distinguish requested AOI area from padded processing grid extent
    left, bottom, right, top = raster_res["bounds"]
    unpadded_w = max(0.0, (right - 200.0) - (left + 200.0))
    unpadded_h = max(0.0, (top - 200.0) - (bottom + 200.0))
    requested_aoi_area_sqkm = round((unpadded_w * unpadded_h) / 1e6, 2)
    padded_grid_area_sqkm = round((raster_res["total_pixel_count"] * (target_res ** 2)) / 1e6, 2)

    # 7. Step 6: Comprehensive Manifest Generation
    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "inputs": {
            "aoi_bbox_wgs84": bbox,
            "event_date": event_date,
            "target_resolution_meters": target_res,
        },
        "timings_seconds": timings,
        "satellite_scenes": {
            "collection": "sentinel-1-rtc",
            "provider": "Microsoft Planetary Computer STAC",
            "relative_orbit": pair_meta["orbit"],
            "orbit_state": pair_meta["orbit_state"],
            "pre_scene": {
                "id": pair_meta["pre_scene"]["id"],
                "datetime": pair_meta["pre_datetime"],
                "coverage_pct": pair_meta["pre_coverage_pct"],
            },
            "post_scene": {
                "id": pair_meta["post_scene"]["id"],
                "datetime": pair_meta["post_datetime"],
                "coverage_pct": pair_meta["post_coverage_pct"],
            },
            "temporal_baseline_days": (
                datetime.fromisoformat(pair_meta["post_datetime"].replace("Z", "+00:00")).date()
                - datetime.fromisoformat(pair_meta["pre_datetime"].replace("Z", "+00:00")).date()
            ).days,
        },
        "dem_source": {
            "collection": "cop-dem-glo-30",
            "resolution": "30m native (resampled to 10m SAR analysis grid)",
            "tiles_used": [d["id"] for d in dem_meta_list],
            "slope_calculation": "2nd-order central differences on 10m projected metric grid",
        },
        "model_details": {
            "architecture": "SNUNet_ECAM (Siamese U-Net++ with Ensemble Channel Attention)",
            "checkpoint": str(checkpoint_path),
            "device": str(engine.device),
            "input_channels": ["Normalized Linear VV", "Normalized Linear VH", "Normalized Copernicus DEM"],
            "patch_size": patch_size,
            "stride": stride,
            "overlap_pixels": patch_size - stride,
            "blending": "2D Hann window weighted stitching",
            "class_mapping": CLASS_NAMES,
            "preprocessing": {
                "input_representation": "Linear radar backscatter power",
                "vv_mean": IMG_MEANS[0],
                "vv_std": IMG_STDS[0],
                "vh_mean": IMG_MEANS[1],
                "vh_std": IMG_STDS[1],
                "dem_mean": DEM_MEAN,
                "dem_std": DEM_STD,
            },
            "domain_gap_disclosure": (
                "Kuro Siwo reference training data used SNAP-preprocessed Sentinel-1 GRD imagery. "
                "Planetary Computer RTC products provide radiometric terrain-corrected gamma-naught measurements. "
                "Inputs were verified as positive linear power and normalized using the official Kuro Siwo statistics."
            ),
        },
        "classical_baseline": {
            "formula": "delta_VV_dB = 10*log10(post_VV) - 10*log10(pre_VV)",
            "threshold_db": db_drop_threshold,
            "max_slope_degrees": max_slope_degrees,
            "stats": base_res["stats"],
            "caveat": "Indicates radar backscatter drop consistent with water smoothing; serves as an unlearned physical reference.",
        },
        "summary_statistics": {
            "grid_dimensions": {"width": raster_res["width"], "height": raster_res["height"]},
            "projected_crs": str(master_crs),
            "total_pixels": raster_res["total_pixel_count"],
            "valid_pixels": raster_res["valid_pixel_count"],
            "valid_fraction": raster_res["valid_fraction"],
            "requested_aoi_area_sqkm": requested_aoi_area_sqkm,
            "padded_grid_area_sqkm": padded_grid_area_sqkm,
            "total_analyzed_area_sqkm": requested_aoi_area_sqkm,
            "analysis_footprint": {
                "user_requested_aoi_bbox": bbox,
                "user_requested_aoi_area_sqkm": requested_aoi_area_sqkm,
                "valid_analyzed_aoi_area_sqkm": requested_aoi_area_sqkm,
                "padded_processing_grid_area_sqkm": padded_grid_area_sqkm,
                "processing_padding_meters": 200.0,
                "padding_disclosure": (
                    "Raster processing uses a 200m peripheral buffer to ensure valid spatial convolution and "
                    "2nd-order central difference slope computation. Headline area statistics exclude processing padding."
                ),
            },
            "snunet_flood_water": {
                "polygon_count": snunet_geojson["properties"]["total_features"],
                "area_sqkm": snunet_geojson["properties"]["total_area_sqkm"],
                "area_hectares": snunet_geojson["properties"]["total_area_hectares"],
            },
            "snunet_permanent_water": {
                "polygon_count": snunet_perm_geojson["properties"]["total_features"],
                "area_sqkm": snunet_perm_geojson["properties"]["total_area_sqkm"],
                "area_hectares": snunet_perm_geojson["properties"]["total_area_hectares"],
            },
            "classical_baseline_flood": {
                "polygon_count": baseline_geojson["properties"]["total_features"],
                "area_sqkm": baseline_geojson["properties"]["total_area_sqkm"],
                "area_hectares": baseline_geojson["properties"]["total_area_hectares"],
            },
        },
        "outputs": {
            "rasters": {k: str(p) for k, p in raster_res["paths"].items()},
            "snunet_flood_prob_tif": str(prob_path),
            "snunet_pred_class_tif": str(class_path),
            "delta_vv_db_tif": str(delta_db_path),
            "baseline_candidate_flood_tif": str(baseline_candidate_path),
            "vectors": {
                "snunet_flood_extent_geojson": str(snunet_geojson_path),
                "snunet_permanent_water_geojson": str(snunet_perm_geojson_path),
                "baseline_candidate_flood_geojson": str(baseline_geojson_path),
            },
        },
        "attribution": {
            "copernicus_sentinel": "Contains modified Copernicus Sentinel data 2024/2026.",
            "copernicus_dem": "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
            "osm": "© OpenStreetMap contributors.",
            "model": "Kuro Siwo: A global multi-temporal SAR dataset for rapid flood mapping (Bountos et al., NeurIPS 2024)",
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 8. Step 7: Optional Infrastructure & Connectivity Analysis
    if run_infrastructure:
        run_infrastructure_and_connectivity_analysis(
            run_dir=output_dir,
            osm_snapshot_timestamp=HISTORICAL_CUTOFF_TIMESTAMP,
        )
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return manifest


def run_infrastructure_and_connectivity_analysis(
    run_dir: Union[str, Path],
    osm_snapshot_timestamp: str = HISTORICAL_CUTOFF_TIMESTAMP,
    max_snapping_distance_m: float = 1500.0,
) -> Dict[str, Any]:
    """
    Executes Phase 3 Historical OSM extraction, infrastructure exposure assessment,
    and road connectivity reachability analysis for an existing analysis run.
    Updates the run's manifest.json and derived/ directory.
    """
    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found in {run_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    bbox = manifest.get("inputs", {}).get("aoi_bbox_wgs84")
    if not bbox:
        raise ValueError(f"aoi_bbox_wgs84 missing from run manifest in {run_path}")

    vectors_dir = run_path / "vectors"
    osm_dir = run_path / "osm"
    derived_dir = run_path / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load flood vector extents
    snunet_path = vectors_dir / "snunet_flood_extent.geojson"
    baseline_path = vectors_dir / "baseline_candidate_flood.geojson"

    snunet_gj = {"type": "FeatureCollection", "features": []}
    if snunet_path.is_file():
        with open(snunet_path, "r", encoding="utf-8") as f:
            snunet_gj = json.load(f)

    baseline_gj = None
    if baseline_path.is_file():
        with open(baseline_path, "r", encoding="utf-8") as f:
            baseline_gj = json.load(f)

    # 2. Extract historical pre-event OSM assets
    osm_data = fetch_historical_osm_data(
        bbox=bbox,
        snapshot_timestamp=osm_snapshot_timestamp,
        cache_dir=osm_dir,
    )

    # 3. Assess infrastructure exposure
    impact_data = assess_infrastructure_impact(
        osm_data=osm_data,
        snunet_flood_geojson=snunet_gj,
        baseline_change_geojson=baseline_gj,
        output_dir=derived_dir,
    )

    # 4. Run road network graph connectivity and reachability analysis
    conn_data = run_connectivity_pipeline(
        osm_data=osm_data,
        impact_data=impact_data,
        output_dir=derived_dir,
        max_snapping_distance_m=max_snapping_distance_m,
    )

    # 5. Update and write manifest
    manifest["infrastructure_impact"] = {
        "snapshot_timestamp": osm_snapshot_timestamp,
        "primary_scenario": impact_data["primary_scenario"],
        "sensitivity_scenario": impact_data.get("sensitivity_scenario"),
        "osm_asset_counts": osm_data.get("counts", {}),
    }
    manifest["connectivity"] = {
        "primary_scenario": conn_data["primary_scenario"],
    }
    if "outputs" not in manifest:
        manifest["outputs"] = {}
    manifest["outputs"]["osm"] = {
        f"osm_{k}_geojson": str(osm_dir / f"osm_{k}.geojson")
        for k in ["roads", "buildings", "bridges", "settlements", "destinations"]
        if (osm_dir / f"osm_{k}.geojson").is_file()
    }
    manifest["outputs"]["derived"] = {
        "asset_impact_geojson": str(derived_dir / "asset_impact.geojson"),
        "potentially_affected_roads_geojson": str(derived_dir / "potentially_affected_roads.geojson"),
        "settlement_accessibility_geojson": str(derived_dir / "settlement_accessibility.geojson"),
        "connectivity_summary_json": str(derived_dir / "connectivity_summary.json"),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "run_id": manifest["run_id"],
        "status": "COMPLETED",
        "infrastructure_impact": manifest["infrastructure_impact"],
        "connectivity": manifest["connectivity"],
        "outputs": manifest["outputs"]["derived"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flood Intelligence Platform - SNUNet & Geospatial Pipeline Runner")
    parser.add_argument("--bbox", nargs=4, type=float, default=[85.15, 27.95, 85.25, 28.05], help="AOI bbox: minx miny maxx maxy")
    parser.add_argument("--date", type=str, default="2026-08-26", help="Flood event date (YYYY-MM-DD)")
    parser.add_argument("--preferred-orbit", type=int, default=19, help="Preferred relative orbit (default: 19)")
    parser.add_argument("--preferred-state", type=str, default="descending", help="Preferred orbit state (default: descending)")
    parser.add_argument("--window", type=int, default=25, help="Search window days (default: 25)")
    parser.add_argument("--threshold-db", type=float, default=-3.0, help="Classical dB drop threshold")
    parser.add_argument("--max-slope", type=float, default=15.0, help="Max slope filter degrees")
    parser.add_argument("--outdir", type=str, default=None, help="Output base directory")

    args = parser.parse_args()
    print("=" * 65)
    print("FLOOD INTELLIGENCE PIPELINE — CLI EXECUTION")
    print("=" * 65)
    print(f"AOI Bounding Box: {args.bbox}")
    print(f"Target Event Date: {args.date}")
    print(f"Preferred Orbit: {args.preferred_orbit} ({args.preferred_state})")

    res = run_flood_analysis(
        aoi=args.bbox,
        event_date=args.date,
        search_window_days=args.window,
        preferred_orbit=args.preferred_orbit,
        preferred_orbit_state=args.preferred_state,
        db_drop_threshold=args.threshold_db,
        max_slope_degrees=args.max_slope,
        output_base_dir=Path(args.outdir) if args.outdir else None,
    )

    print("\nEXECUTION SUCCESSFUL!")
    print(f"Run ID: {res['run_id']}")
    print(f"Pre-Scene:  {res['satellite_scenes']['pre_scene']['id']}")
    print(f"Post-Scene: {res['satellite_scenes']['post_scene']['id']}")
    print(f"Analyzed Area: {res['summary_statistics']['total_analyzed_area_sqkm']} km²")
    print(f"SNUNet Flood Area: {res['summary_statistics']['snunet_flood_water']['area_sqkm']} km²")
    print(f"Classical Flood Area: {res['summary_statistics']['classical_baseline_flood']['area_sqkm']} km²")
    print(f"Total Duration: {res['timings_seconds']['total_pipeline_seconds']} seconds")
    print(f"Outputs written to: runs/{res['run_id']}")
