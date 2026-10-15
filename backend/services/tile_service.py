"""
Raster Tile Rendering Service
Renders georeferenced 256x256 Web Mercator (EPSG:3857) PNG raster tiles
directly from analysis GeoTIFFs using Rasterio and Pillow.
Supports secure layer allowlisting, physical dB scaling, and LRU memory caching.
"""

import io
import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Set
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent

# Allowed raster layers and corresponding file basenames under runs/<run_id>/rasters/
ALLOWED_LAYERS: Dict[str, str] = {
    "pre_vv": "pre_vv.tif",
    "post_vv": "post_vv.tif",
    "pre_vh": "pre_vh.tif",
    "post_vh": "post_vh.tif",
    "delta_vv_db": "delta_vv_db.tif",
    "snunet_flood_prob": "snunet_flood_prob.tif",
    "snunet_pred_class": "snunet_pred_class.tif",
    "baseline_candidate_flood": "baseline_candidate_flood.tif",
    "dem": "dem.tif",
    "slope": "slope.tif",
}

# Web Mercator constant
ORIGIN_SHIFT = 20037508.342789244
EMPTY_TILE_BYTES = None


def get_empty_tile_png() -> bytes:
    """Returns a cached 256x256 transparent PNG."""
    global EMPTY_TILE_BYTES
    if EMPTY_TILE_BYTES is None:
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        EMPTY_TILE_BYTES = buf.getvalue()
    return EMPTY_TILE_BYTES


def xyz_to_epsg3857_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """
    Computes Web Mercator (EPSG:3857) metric bounds [min_x, min_y, max_x, max_y]
    for XYZ tile coordinates.
    """
    n = 2.0 ** z
    tile_size = 2.0 * ORIGIN_SHIFT / n
    min_x = -ORIGIN_SHIFT + x * tile_size
    max_x = -ORIGIN_SHIFT + (x + 1) * tile_size
    max_y = ORIGIN_SHIFT - y * tile_size
    min_y = ORIGIN_SHIFT - (y + 1) * tile_size
    return min_x, min_y, max_x, max_y


def colorize_sar_linear(data: np.ndarray, nodata_mask: np.ndarray, vmin_db: float = -25.0, vmax_db: float = -3.0) -> np.ndarray:
    """
    Colorizes linear SAR backscatter into calibrated grayscale RGBA tile.
    Uses identical dB scaling for pre and post imagery to ensure valid visual comparison.
    """
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    valid = (~nodata_mask) & (data > 1e-7)

    if np.any(valid):
        db = 10.0 * np.log10(np.maximum(data[valid], 1e-7))
        norm = np.clip((db - vmin_db) / (vmax_db - vmin_db), 0.0, 1.0)
        gray = (norm * 255.0).astype(np.uint8)

        rgba[valid, 0] = gray
        rgba[valid, 1] = gray
        rgba[valid, 2] = gray
        rgba[valid, 3] = 255

    return rgba


def colorize_delta_db(data: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """
    Colorizes radar backscatter drop (water/smoothing) vs increase.
    Drops < -3 dB rendered as electric blue; increases > +3 dB rendered as coral red.
    """
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    valid = (~nodata_mask) & (data != 0.0)

    # Significant drop: water candidate
    drop_mask = valid & (data <= -3.0)
    if np.any(drop_mask):
        intensity = np.clip((-data[drop_mask] - 3.0) / 7.0, 0.0, 1.0)
        alpha = (140 + intensity * 100).astype(np.uint8)
        rgba[drop_mask, 0] = 30
        rgba[drop_mask, 1] = 120
        rgba[drop_mask, 2] = 255
        rgba[drop_mask, 3] = alpha

    # Significant increase
    inc_mask = valid & (data >= 3.0)
    if np.any(inc_mask):
        intensity = np.clip((data[inc_mask] - 3.0) / 7.0, 0.0, 1.0)
        alpha = (120 + intensity * 100).astype(np.uint8)
        rgba[inc_mask, 0] = 240
        rgba[inc_mask, 1] = 80
        rgba[inc_mask, 2] = 80
        rgba[inc_mask, 3] = alpha

    return rgba


def colorize_snunet_prob(data: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """Colorizes SNUNet flood probability from 0.0 to 1.0."""
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    valid = (~nodata_mask) & (data >= 0.2)

    if np.any(valid):
        prob = data[valid]
        rgba[valid, 0] = (20 + (1.0 - prob) * 30).astype(np.uint8)
        rgba[valid, 1] = (100 + prob * 140).astype(np.uint8)
        rgba[valid, 2] = 255
        rgba[valid, 3] = (120 + prob * 130).astype(np.uint8)

    return rgba


def colorize_snunet_class(data: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """
    Colorizes SNUNet predicted classes:
    0: Background (Transparent)
    1: Permanent Water (Turquoise Cyan)
    2: Flood Water (Electric Blue)
    """
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    classes = np.round(data).astype(np.uint8)

    # Permanent Water (Class 1)
    perm_mask = (~nodata_mask) & (classes == 1)
    rgba[perm_mask, 0] = 6    # #06b6d4
    rgba[perm_mask, 1] = 182
    rgba[perm_mask, 2] = 212
    rgba[perm_mask, 3] = 180

    # Flood Water (Class 2)
    flood_mask = (~nodata_mask) & (classes == 2)
    rgba[flood_mask, 0] = 59   # #3b82f6
    rgba[flood_mask, 1] = 130
    rgba[flood_mask, 2] = 246
    rgba[flood_mask, 3] = 220

    return rgba


def colorize_baseline_candidate(data: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """Colorizes classical radar change candidate flood mask (Amber / Orange)."""
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    mask = (~nodata_mask) & (np.round(data).astype(np.uint8) == 1)

    rgba[mask, 0] = 245  # #f59e0b
    rgba[mask, 1] = 158
    rgba[mask, 2] = 11
    rgba[mask, 3] = 175

    return rgba


def colorize_dem(data: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """Colorizes Copernicus DEM elevation (500m to 3500m in Trishuli region)."""
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    valid = (~nodata_mask) & (data > 0.0)

    if np.any(valid):
        norm = np.clip((data[valid] - 400.0) / 2800.0, 0.0, 1.0)
        # Terrain hypsometric tint (green -> yellow -> brown -> white)
        rgba[valid, 0] = (80 + norm * 160).astype(np.uint8)
        rgba[valid, 1] = (160 - norm * 60).astype(np.uint8)
        rgba[valid, 2] = (80 + norm * 150).astype(np.uint8)
        rgba[valid, 3] = 190

    return rgba


def colorize_slope(data: np.ndarray, nodata_mask: np.ndarray) -> np.ndarray:
    """Colorizes terrain slope in degrees (0 to 60 deg)."""
    rgba = np.zeros((256, 256, 4), dtype=np.uint8)
    valid = (~nodata_mask) & (data >= 0.0)

    if np.any(valid):
        norm = np.clip(data[valid] / 45.0, 0.0, 1.0)
        # Gentle (green) to Steep (red)
        rgba[valid, 0] = (norm * 240).astype(np.uint8)
        rgba[valid, 1] = ((1.0 - norm) * 200).astype(np.uint8)
        rgba[valid, 2] = 40
        rgba[valid, 3] = 160

    return rgba


@lru_cache(maxsize=1024)
def render_raster_tile(
    run_id: str,
    layer_name: str,
    z: int,
    x: int,
    y: int,
) -> bytes:
    """
    Renders a 256x256 Web Mercator PNG tile for a given run ID and layer.
    Cached in memory via LRU cache.
    """
    if layer_name not in ALLOWED_LAYERS:
        raise ValueError(f"Layer '{layer_name}' not in allowed tile layers: {list(ALLOWED_LAYERS.keys())}")

    run_dir = WORKSPACE_ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Analysis run '{run_id}' not found.")

    tif_filename = ALLOWED_LAYERS[layer_name]
    tif_path = run_dir / "rasters" / tif_filename
    if not tif_path.is_file():
        raise FileNotFoundError(f"Raster '{tif_filename}' not found for run '{run_id}'.")

    min_x, min_y, max_x, max_y = xyz_to_epsg3857_bounds(z, x, y)

    with rasterio.open(tif_path) as src:
        # Reproject src bounds to EPSG:3857 to test intersection
        src_left, src_bottom, src_right, src_top = transform_bounds(
            src.crs, "EPSG:3857", *src.bounds
        )

        # Check if tile bounds intersect raster bounds
        intersects = not (
            max_x < src_left or min_x > src_right or max_y < src_bottom or min_y > src_top
        )
        if not intersects:
            return get_empty_tile_png()

        # Destination 256x256 Web Mercator grid
        res_x = (max_x - min_x) / 256.0
        res_y = (max_y - min_y) / 256.0
        dst_transform = rasterio.Affine(res_x, 0.0, min_x, 0.0, -res_y, max_y)

        nodata_val = src.nodata if src.nodata is not None else -9999.0
        dst_array = np.full((256, 256), fill_value=nodata_val, dtype=np.float32)

        resampling_type = Resampling.nearest if layer_name in ["snunet_pred_class", "baseline_candidate_flood"] else Resampling.bilinear

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs="EPSG:3857",
            dst_nodata=nodata_val,
            resampling=resampling_type,
        )

    nodata_mask = np.isnan(dst_array) | (dst_array == nodata_val) | (dst_array <= -9000.0)

    # Colorize according to layer type
    if layer_name in ["pre_vv", "post_vv"]:
        rgba = colorize_sar_linear(dst_array, nodata_mask, vmin_db=-25.0, vmax_db=-3.0)
    elif layer_name in ["pre_vh", "post_vh"]:
        rgba = colorize_sar_linear(dst_array, nodata_mask, vmin_db=-30.0, vmax_db=-8.0)
    elif layer_name == "delta_vv_db":
        rgba = colorize_delta_db(dst_array, nodata_mask)
    elif layer_name == "snunet_flood_prob":
        rgba = colorize_snunet_prob(dst_array, nodata_mask)
    elif layer_name == "snunet_pred_class":
        rgba = colorize_snunet_class(dst_array, nodata_mask)
    elif layer_name == "baseline_candidate_flood":
        rgba = colorize_baseline_candidate(dst_array, nodata_mask)
    elif layer_name == "dem":
        rgba = colorize_dem(dst_array, nodata_mask)
    elif layer_name == "slope":
        rgba = colorize_slope(dst_array, nodata_mask)
    else:
        rgba = np.zeros((256, 256, 4), dtype=np.uint8)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
