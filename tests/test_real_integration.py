"""
Phase 2 Integration Test — Real Satellite Data & End-to-End Pipeline
Executes genuine Microsoft Planetary Computer Sentinel-1 RTC queries, Copernicus DEM
alignment, GPU-accelerated SNUNet-ECAM inference, classical radar change detection,
and GeoJSON flood extent generation on the Bhote Koshi–Trishuli River Corridor.
"""

import sys
import json
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import rasterio
from backend.pipeline import run_flood_analysis


from backend.services.stac_service import select_best_scene_pair


def test_real_trishuli_2026_challenge_case_study():
    """
    Executes the genuine 2026 Trishuli flood analysis pipeline using real Sentinel-1
    observations from August/September 2026 and Copernicus DEM.
    Validates all spatial rasters, GeoJSON vectors, and manifest metadata.
    """
    # Trishuli River corridor focus AOI (~10 km x 10 km)
    trishuli_bbox = [85.15, 27.95, 85.25, 28.05]
    event_date = "2026-08-26"

    manifest = run_flood_analysis(
        aoi=trishuli_bbox,
        event_date=event_date,
        preferred_orbit=19,
        preferred_orbit_state="descending",
        search_window_days=25,
        min_aoi_coverage_pct=70.0,
        db_drop_threshold=-3.0,
        max_slope_degrees=15.0,
        batch_size=4,
    )

    # 1. Validate Manifest Completeness and 2026 Event Date
    assert manifest["status"] == "COMPLETED"
    assert manifest["run_id"].startswith("flood_")
    assert manifest["inputs"]["event_date"] == "2026-08-26", "Must be 2026-08-26 case study"
    assert "satellite_scenes" in manifest
    assert "model_details" in manifest
    assert "summary_statistics" in manifest
    assert "outputs" in manifest

    scenes = manifest["satellite_scenes"]
    print(f"\n[Real 2026 STAC Pair]")
    print(f"  Pre-Scene:  {scenes['pre_scene']['id']} ({scenes['pre_scene']['datetime']})")
    print(f"  Post-Scene: {scenes['post_scene']['id']} ({scenes['post_scene']['datetime']})")
    print(f"  Orbit:      {scenes['relative_orbit']} ({scenes['orbit_state']})")
    print(f"  Baseline:   {scenes['temporal_baseline_days']} days")

    assert scenes["pre_scene"]["id"] != scenes["post_scene"]["id"], "Pre and Post scenes must be different"
    assert scenes["pre_scene"]["datetime"].startswith("2026-08-24"), "Pre-scene must be 24 Aug 2026"
    assert scenes["post_scene"]["datetime"].startswith("2026-09-05"), "Post-scene must be 5 Sep 2026"
    assert scenes["relative_orbit"] == 19, f"Expected Orbit 19, got {scenes['relative_orbit']}"
    assert scenes["orbit_state"] == "descending", f"Expected descending, got {scenes['orbit_state']}"
    assert scenes["temporal_baseline_days"] == 12, "Must be 12-day repeat orbit baseline"

    # 2. Validate Generated GeoTIFF Outputs
    rasters = manifest["outputs"]["rasters"]
    required_rasters = ["pre_vv", "pre_vh", "post_vv", "post_vh", "dem", "slope", "valid_mask"]
    for rname in required_rasters:
        rpath = Path(rasters[rname])
        assert rpath.is_file(), f"Raster output missing: {rpath}"
        with rasterio.open(rpath) as src:
            assert src.crs is not None, f"CRS missing in {rname}"
            assert src.transform is not None, f"Transform missing in {rname}"
            assert src.shape[0] > 100 and src.shape[1] > 100, f"Unexpected grid shape in {rname}"

    # Validate Model Output Rasters
    prob_path = Path(manifest["outputs"]["snunet_flood_prob_tif"])
    class_path = Path(manifest["outputs"]["snunet_pred_class_tif"])
    delta_path = Path(manifest["outputs"]["delta_vv_db_tif"])
    assert prob_path.is_file() and class_path.is_file() and delta_path.is_file()

    with rasterio.open(prob_path) as src:
        prob_data = src.read(1)
        assert (prob_data >= 0.0).all() and (prob_data <= 1.0).all(), "Probabilities must be in [0, 1]"

    with rasterio.open(class_path) as src:
        class_data = src.read(1)
        unique_classes = set(class_data.flatten().tolist())
        assert unique_classes.issubset({0, 1, 2}), f"Unexpected classes: {unique_classes}"

    # 3. Validate GeoJSON Vector Outputs
    vectors = manifest["outputs"]["vectors"]
    flood_geojson_path = Path(vectors["snunet_flood_extent_geojson"])
    baseline_geojson_path = Path(vectors["baseline_candidate_flood_geojson"])
    assert flood_geojson_path.is_file() and baseline_geojson_path.is_file()

    with open(flood_geojson_path, "r", encoding="utf-8") as f:
        flood_gj = json.load(f)
    assert flood_gj["type"] == "FeatureCollection"
    assert "total_area_sqkm" in flood_gj["properties"]

    stats = manifest["summary_statistics"]
    print(f"\n[2026 Summary Statistics]")
    print(f"  Grid Size:           {stats['grid_dimensions']['width']} x {stats['grid_dimensions']['height']}")
    print(f"  Analyzed Area:       {stats['total_analyzed_area_sqkm']} km²")
    print(f"  SNUNet Flood Area:   {stats['snunet_flood_water']['area_sqkm']} km² ({stats['snunet_flood_water']['polygon_count']} polygons)")
    print(f"  SNUNet Perm. Water:  {stats['snunet_permanent_water']['area_sqkm']} km²")
    print(f"  Baseline Flood Area: {stats['classical_baseline_flood']['area_sqkm']} km²")
    print(f"  Total Run Time:      {manifest['timings_seconds']['total_pipeline_seconds']} seconds")

    print("\n2026 REAL CHALLENGE CASE STUDY TEST PASSED SUCCESSFULLY!")


def test_real_trishuli_2024_historical_regression_pairing():
    """
    Historical regression test verifying that 2024 historical observations can still be
    queried and identified separately from the 2026 challenge case study.
    """
    trishuli_bbox = [85.15, 27.95, 85.25, 28.05]
    pair = select_best_scene_pair(
        aoi=trishuli_bbox,
        event_date_str="2024-08-26",
        preferred_orbit=19,
        preferred_orbit_state="descending",
    )
    assert pair["orbit"] == 19
    assert pair["pre_datetime"].startswith("2024-08-15")
    assert pair["post_datetime"].startswith("2024-08-27")
    assert "S1A_IW_GRDH" in pair["pre_scene"]["id"]
    print(f"\n[Historical 2024 Regression Pair Verified: {pair['pre_scene']['id']} / {pair['post_scene']['id']}]")


if __name__ == "__main__":
    test_real_trishuli_2026_challenge_case_study()
    test_real_trishuli_2024_historical_regression_pairing()
