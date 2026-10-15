"""
Phase 2 Unit Tests — Geospatial & Algorithmic Pipeline Subsystems
Tests grid alignment, physical slope calculation, decibel backscatter difference,
classical change detection baseline, vector polygonization, and empty result handling.
"""

import sys
from pathlib import Path
import numpy as np
import rasterio

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.services.raster_service import (
    compute_target_grid,
    compute_slope_degrees,
)
from backend.services.baseline_service import (
    compute_radar_change_baseline,
    BaselineConfig,
)
from backend.services.polygon_service import (
    mask_to_geojson_features,
)
from backend.services.stac_service import (
    NoCompatibleScenePairError,
    aoi_to_bbox_and_geom,
)


def test_grid_computation_and_alignment():
    """Verifies that bounding boxes are converted and snapped to 10m grid."""
    sar_crs = rasterio.crs.CRS.from_epsg(32645)  # UTM 45N
    bbox_wgs84 = [85.1, 27.9, 85.3, 28.1]

    transform, width, height, bounds = compute_target_grid(
        aoi_bbox_wgs84=bbox_wgs84,
        sar_src_crs=sar_crs,
        target_res=10.0,
        buffer_meters=100.0,
    )

    left, bottom, right, top = bounds
    assert left % 10.0 == 0.0, "Left boundary not aligned to 10m"
    assert bottom % 10.0 == 0.0, "Bottom boundary not aligned to 10m"
    assert right % 10.0 == 0.0, "Right boundary not aligned to 10m"
    assert top % 10.0 == 0.0, "Top boundary not aligned to 10m"
    assert width == int((right - left) / 10.0)
    assert height == int((top - bottom) / 10.0)
    assert transform.a == 10.0 and transform.e == -10.0


def test_slope_computation_physical_units():
    """Verifies central difference slope calculation in degrees on a metric grid."""
    # Create synthetic flat plane
    flat = np.full((100, 100), 1000.0, dtype=np.float32)
    slope_flat = compute_slope_degrees(flat, res_meters=10.0)
    assert np.allclose(slope_flat[1:-1, 1:-1], 0.0, atol=1e-5), "Flat surface must have 0 deg slope"

    # Create 45-degree ramp in X direction: dz/dx = 1.0 (rise = run)
    x = np.arange(100) * 10.0  # 10m step
    ramp = np.tile(x, (100, 1)).astype(np.float32)
    slope_ramp = compute_slope_degrees(ramp, res_meters=10.0)
    # arctan(1.0) = 45 degrees
    assert np.allclose(slope_ramp[1:-1, 1:-1], 45.0, atol=1e-2), f"Expected 45 deg slope, got {slope_ramp[1:-1, 1:-1].mean()}"


def test_decibel_conversion_and_baseline_logic():
    """Verifies decibel formula: delta_VV_dB = 10*log10(post) - 10*log10(pre)."""
    # pre VV = 0.10 (-10 dB), post VV = 0.01 (-20 dB) -> delta = -10 dB (drop)
    pre = np.full((50, 50), 0.10, dtype=np.float32)
    post = np.full((50, 50), 0.01, dtype=np.float32)
    slope = np.full((50, 50), 5.0, dtype=np.float32)  # 5 deg < 15 deg
    valid = np.ones((50, 50), dtype=bool)

    cfg = BaselineConfig(db_drop_threshold=-3.0, max_slope_degrees=15.0)
    res = compute_radar_change_baseline(pre, post, slope, valid, config=cfg)

    # -20 - (-10) = -10 dB
    assert np.allclose(res["delta_vv_db"], -10.0, atol=0.1)
    assert np.all(res["raw_change_mask"] == 1), "Drop of -10 dB should trigger raw change (< -3 dB)"
    assert np.all(res["candidate_flood_mask"] == 1), "Slope 5 deg < 15 deg should retain candidate"

    # Test slope rejection on steep terrain (> 15 deg)
    steep_slope = np.full((50, 50), 30.0, dtype=np.float32)
    res_steep = compute_radar_change_baseline(pre, post, steep_slope, valid, config=cfg)
    assert np.all(res_steep["raw_change_mask"] == 1), "Raw drop still detected"
    assert np.all(res_steep["candidate_flood_mask"] == 0), "Steep slope (30 deg) must filter out candidate flood"


def test_polygonization_and_empty_prediction():
    """Verifies vector polygon extraction, metric area calculation, and empty handling."""
    crs = rasterio.crs.CRS.from_epsg(32645)  # UTM 45N
    transform = rasterio.Affine(10.0, 0.0, 300000.0, 0.0, -10.0, 3100000.0)

    # 1. Empty mask case (critical constraint: no fabricated polygons)
    empty_mask = np.zeros((100, 100), dtype=np.uint8)
    empty_geojson = mask_to_geojson_features(empty_mask, transform, crs, target_value=1)
    assert empty_geojson["type"] == "FeatureCollection"
    assert len(empty_geojson["features"]) == 0
    assert empty_geojson["properties"]["total_area_sqkm"] == 0.0

    # 2. Non-empty mask with known square: 10 x 10 pixels = 100m x 100m = 10,000 m² = 1 hectare = 0.01 km²
    square_mask = np.zeros((100, 100), dtype=np.uint8)
    square_mask[20:30, 20:30] = 1
    geojson = mask_to_geojson_features(square_mask, transform, crs, target_value=1, min_area_sqm=500.0)

    assert len(geojson["features"]) == 1
    feat = geojson["features"][0]
    assert abs(feat["properties"]["area_sqm"] - 10000.0) < 100.0
    assert abs(feat["properties"]["area_hectares"] - 1.0) < 0.02
    assert abs(feat["properties"]["area_sqkm"] - 0.01) < 0.001

    # Verify coordinates are in WGS 84 (Nepal lon ~85, lat ~28)
    coords = feat["geometry"]["coordinates"][0][0]
    lon, lat = coords[0], coords[1]
    assert 80.0 < lon < 90.0, f"Expected longitude in Nepal, got {lon}"
    assert 20.0 < lat < 35.0, f"Expected latitude in Nepal, got {lat}"


def test_aoi_format_validation():
    """Verifies that bbox and GeoJSON formats are validated correctly."""
    bbox, geom = aoi_to_bbox_and_geom([85.1, 27.9, 85.3, 28.1])
    assert len(bbox) == 4
    assert geom.area > 0

    # GeoJSON dict
    geojson_poly = {
        "type": "Polygon",
        "coordinates": [[[85.1, 27.9], [85.3, 27.9], [85.3, 28.1], [85.1, 28.1], [85.1, 27.9]]],
    }
    bbox2, geom2 = aoi_to_bbox_and_geom(geojson_poly)
    assert abs(bbox2[0] - 85.1) < 1e-4
    assert geom2.area > 0


if __name__ == "__main__":
    test_grid_computation_and_alignment()
    test_slope_computation_physical_units()
    test_decibel_conversion_and_baseline_logic()
    test_polygonization_and_empty_prediction()
    test_aoi_format_validation()
    print("ALL GEOSPATIAL PIPELINE UNIT TESTS PASSED!")
