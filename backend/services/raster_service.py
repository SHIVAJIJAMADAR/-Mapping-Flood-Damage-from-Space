"""
Raster Extraction and Geospatial Alignment Service
Extracts georeferenced windows for Sentinel-1 RTC (VV, VH) and Copernicus DEM,
aligns them to a shared metric UTM grid, computes physical slope, and builds valid masks.
"""

import math
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.enums import Resampling as RioResampling
from pyproj import Transformer
from shapely.geometry import box

# Optimal GDAL configuration for windowed Cloud Optimized GeoTIFF reads over Azure Blob Storage SAS URLs
GDAL_COG_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "5",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "VSI_CACHE": "TRUE",
}


class RasterExtractionError(Exception):
    """Raised when raster extraction or alignment fails."""
    pass


def compute_target_grid(
    aoi_bbox_wgs84: List[float],
    sar_src_crs: rasterio.crs.CRS,
    target_res: float = 10.0,
    buffer_meters: float = 200.0,
) -> Tuple[rasterio.Affine, int, int, Tuple[float, float, float, float]]:
    """
    Transforms WGS 84 bbox [minx, miny, maxx, maxy] into projected SAR coordinates
    and aligns the bounding box to exact integer multiples of target_res.

    Returns:
      (transform, width, height, (left, bottom, right, top))
    """
    transformer = Transformer.from_crs("EPSG:4326", sar_src_crs, always_xy=True)
    min_x, min_y = transformer.transform(aoi_bbox_wgs84[0], aoi_bbox_wgs84[1])
    max_x, max_y = transformer.transform(aoi_bbox_wgs84[2], aoi_bbox_wgs84[3])

    # Ensure min/max ordering
    min_x, max_x = min(min_x, max_x) - buffer_meters, max(min_x, max_x) + buffer_meters
    min_y, max_y = min(min_y, max_y) - buffer_meters, max(min_y, max_y) + buffer_meters

    # Align to grid
    left = math.floor(min_x / target_res) * target_res
    bottom = math.floor(min_y / target_res) * target_res
    right = math.ceil(max_x / target_res) * target_res
    top = math.ceil(max_y / target_res) * target_res

    width = int(round((right - left) / target_res))
    height = int(round((top - bottom) / target_res))

    transform = rasterio.Affine(target_res, 0.0, left, 0.0, -target_res, top)
    return transform, width, height, (left, bottom, right, top)


def read_windowed_sar_asset(
    signed_url: str,
    target_crs: rasterio.crs.CRS,
    target_transform: rasterio.Affine,
    width: int,
    height: int,
    bounds: Tuple[float, float, float, float],
    max_retries: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a windowed region from a Sentinel-1 RTC COG using signed URL.
    Returns (data_array, valid_mask_array).
    """
    left, bottom, right, top = bounds
    out_array = np.zeros((height, width), dtype=np.float32)

    last_err = None
    for attempt in range(max_retries):
        try:
            with rasterio.Env(**GDAL_COG_ENV):
                with rasterio.open(signed_url) as src:
                    nodata_val = src.nodata if src.nodata is not None else -32768.0

                    if src.crs == target_crs:
                        # Native projection: extract directly via bounds window
                        win = from_bounds(left, bottom, right, top, transform=src.transform)
                        win_int = win.round_offsets().round_lengths()
                        raw = src.read(1, window=win_int, boundless=True, fill_value=nodata_val)

                        # Handle shape mismatch if round_offsets differed by 1 px
                        if raw.shape == (height, width):
                            out_array = raw
                        else:
                            reproject(
                                source=raw,
                                destination=out_array,
                                src_transform=rasterio.windows.transform(win_int, src.transform),
                                src_crs=src.crs,
                                dst_transform=target_transform,
                                dst_crs=target_crs,
                                resampling=RioResampling.bilinear,
                            )
                    else:
                        # Different CRS: reproject directly into target grid
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=out_array,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=target_crs,
                            dst_nodata=nodata_val,
                            resampling=RioResampling.bilinear,
                        )
            last_err = None
            break
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RasterExtractionError(f"Failed to read windowed SAR asset after {max_retries} attempts: {e}") from e

    # Valid mask: positive, finite, not nodata
    valid_mask = np.isfinite(out_array) & (out_array > 0.0) & (out_array != nodata_val)
    # Zero-out nodata or negative backscatter
    out_array[~valid_mask] = 0.0
    return out_array, valid_mask


def read_aligned_dem(
    dem_urls: List[str],
    target_crs: rasterio.crs.CRS,
    target_transform: rasterio.Affine,
    width: int,
    height: int,
    bounds: Tuple[float, float, float, float],
    max_retries: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mosaics and reprojects Copernicus GLO-30 DEM into the target SAR UTM grid.
    Returns (dem_array, valid_mask).
    """
    dem_out = np.zeros((height, width), dtype=np.float32)
    dem_valid = np.zeros((height, width), dtype=bool)

    if not dem_urls:
        raise RasterExtractionError("No Copernicus DEM URLs provided for extraction.")

    with rasterio.Env(**GDAL_COG_ENV):
        for dem_url in dem_urls:
            tile_out = np.zeros((height, width), dtype=np.float32)
            last_err = None
            for attempt in range(max_retries):
                try:
                    with rasterio.open(dem_url) as src:
                        nodata_val = src.nodata if src.nodata is not None else -9999.0
                        reproject(
                            source=rasterio.band(src, 1),
                            destination=tile_out,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=target_transform,
                            dst_crs=target_crs,
                            dst_nodata=nodata_val,
                            resampling=RioResampling.bilinear,
                        )
                        tile_valid = np.isfinite(tile_out) & (tile_out != nodata_val) & (tile_out > -500.0) & (tile_out < 9000.0)

                        # Merge tiles (where not already filled)
                        unfilled = ~dem_valid & tile_valid
                        dem_out[unfilled] = tile_out[unfilled]
                        dem_valid[unfilled] = True
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < max_retries - 1:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        raise RasterExtractionError(f"Failed to read DEM tile after {max_retries} attempts: {e}") from e

    return dem_out, dem_valid


def compute_slope_degrees(
    dem_meters: np.ndarray,
    res_meters: float = 10.0,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes terrain slope in degrees using 2nd-order central differences on a metric grid.

    slope = arctan(sqrt((dz/dx)^2 + (dz/dy)^2)) * (180 / pi)
    """
    # Gradient in y (axis 0) and x (axis 1)
    dz_dy, dz_dx = np.gradient(dem_meters, res_meters, res_meters)
    slope_rad = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))
    slope_deg = np.rad2deg(slope_rad).astype(np.float32)

    if valid_mask is not None:
        slope_deg[~valid_mask] = 0.0

    return slope_deg


def save_geotiff(
    output_path: Path,
    data: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    nodata: Optional[float] = None,
    dtype: Optional[str] = None,
) -> Path:
    """
    Saves a 2D numpy array as a georeferenced Cloud-Optimized compatible GeoTIFF.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_dtype = dtype if dtype else str(data.dtype)
    height, width = data.shape

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=out_dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(data.astype(out_dtype), 1)

    return output_path


def extract_and_align_rasters(
    pre_signed_urls: Dict[str, str],
    post_signed_urls: Dict[str, str],
    dem_signed_urls: List[str],
    aoi_bbox: List[float],
    output_dir: Path,
    target_res: float = 10.0,
) -> Dict[str, Any]:
    """
    Orchestrates end-to-end extraction, alignment, and slope generation.

    Returns a dict with metadata, grid dimensions, and saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Determine master projection and grid from pre-VV
    with rasterio.Env(**GDAL_COG_ENV):
        with rasterio.open(pre_signed_urls["vv"]) as sample_src:
            master_crs = sample_src.crs

    transform, width, height, bounds = compute_target_grid(
        aoi_bbox_wgs84=aoi_bbox,
        sar_src_crs=master_crs,
        target_res=target_res,
        buffer_meters=200.0,
    )

    # 2. Extract SAR observations
    pre_vv, pre_vv_valid = read_windowed_sar_asset(
        pre_signed_urls["vv"], master_crs, transform, width, height, bounds
    )
    pre_vh, pre_vh_valid = read_windowed_sar_asset(
        pre_signed_urls["vh"], master_crs, transform, width, height, bounds
    )
    post_vv, post_vv_valid = read_windowed_sar_asset(
        post_signed_urls["vv"], master_crs, transform, width, height, bounds
    )
    post_vh, post_vh_valid = read_windowed_sar_asset(
        post_signed_urls["vh"], master_crs, transform, width, height, bounds
    )

    # 3. Extract and align DEM
    dem, dem_valid = read_aligned_dem(
        dem_signed_urls, master_crs, transform, width, height, bounds
    )

    # 4. Compute slope
    slope = compute_slope_degrees(dem, res_meters=target_res, valid_mask=dem_valid)

    # 5. Build composite valid mask
    composite_valid = (
        pre_vv_valid & pre_vh_valid & post_vv_valid & post_vh_valid & dem_valid
    )

    # 6. Save GeoTIFFs to run directory
    paths = {
        "pre_vv": save_geotiff(output_dir / "pre_vv.tif", pre_vv, transform, master_crs, nodata=0.0),
        "pre_vh": save_geotiff(output_dir / "pre_vh.tif", pre_vh, transform, master_crs, nodata=0.0),
        "post_vv": save_geotiff(output_dir / "post_vv.tif", post_vv, transform, master_crs, nodata=0.0),
        "post_vh": save_geotiff(output_dir / "post_vh.tif", post_vh, transform, master_crs, nodata=0.0),
        "dem": save_geotiff(output_dir / "dem.tif", dem, transform, master_crs, nodata=-9999.0),
        "slope": save_geotiff(output_dir / "slope.tif", slope, transform, master_crs, nodata=-9999.0),
        "valid_mask": save_geotiff(output_dir / "valid_mask.tif", composite_valid.astype(np.uint8), transform, master_crs, nodata=0, dtype="uint8"),
    }

    return {
        "crs": str(master_crs),
        "transform": list(transform)[:6],
        "width": width,
        "height": height,
        "bounds": bounds,
        "resolution_m": target_res,
        "valid_pixel_count": int(np.sum(composite_valid)),
        "total_pixel_count": int(width * height),
        "valid_fraction": float(np.mean(composite_valid)),
        "paths": paths,
        "arrays": {
            "pre_vv": pre_vv,
            "pre_vh": pre_vh,
            "post_vv": post_vv,
            "post_vh": post_vh,
            "dem": dem,
            "slope": slope,
            "valid_mask": composite_valid,
        },
    }
