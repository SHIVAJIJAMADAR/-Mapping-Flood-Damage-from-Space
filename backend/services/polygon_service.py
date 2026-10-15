"""
Geospatial Polygonization and Area Statistics Service
Converts binary/categorical flood raster masks to clean GeoJSON FeatureCollections
with projected metric area calculations and WGS 84 coordinate reprojection.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import transform as shapely_transform
from pyproj import Transformer


def mask_to_geojson_features(
    mask_array: np.ndarray,
    transform: rasterio.Affine,
    src_crs: rasterio.crs.CRS,
    target_value: int = 1,
    min_area_sqm: float = 500.0,
) -> Dict[str, Any]:
    """
    Extracts polygons from a binary raster mask, filters speckle noise (< min_area_sqm),
    computes exact projected metric areas (m², hectares, km²), and reprojects to WGS 84.

    Returns a GeoJSON FeatureCollection dict.
    """
    # Binary mask uint8
    bin_mask = (mask_array == target_value).astype(np.uint8)
    if np.sum(bin_mask) == 0:
        return {
            "type": "FeatureCollection",
            "features": [],
            "properties": {
                "total_features": 0,
                "total_area_sqm": 0.0,
                "total_area_hectares": 0.0,
                "total_area_sqkm": 0.0,
            },
        }

    # Coordinate transformer from projected metric CRS to WGS 84
    to_wgs84 = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True).transform

    features = []
    total_area_sqm = 0.0
    feature_id = 0

    # Extract polygon shapes using rasterio.features.shapes
    for geom_dict, val in shapes(bin_mask, mask=bin_mask.astype(bool), transform=transform):
        if int(val) != target_value:
            continue

        poly = shape(geom_dict)
        # Simplify slightly to remove micro-pixel staircase artifacts
        poly = poly.simplify(0.5, preserve_topology=True)
        area_m2 = poly.area

        # Filter out isolated sub-resolution speckle
        if area_m2 < min_area_sqm:
            continue

        total_area_sqm += area_m2
        feature_id += 1

        # Reproject geometry to WGS 84 for standard GeoJSON compatibility
        import shapely
        wgs84_poly = shapely.transform(poly, to_wgs84, interleaved=False)

        features.append({
            "type": "Feature",
            "id": feature_id,
            "geometry": mapping(wgs84_poly),
            "properties": {
                "id": feature_id,
                "class_id": int(target_value),
                "area_sqm": round(area_m2, 1),
                "area_hectares": round(area_m2 / 10000.0, 3),
                "area_sqkm": round(area_m2 / 1000000.0, 4),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "total_features": len(features),
            "total_area_sqm": round(total_area_sqm, 1),
            "total_area_hectares": round(total_area_sqm / 10000.0, 3),
            "total_area_sqkm": round(total_area_sqm / 1000000.0, 4),
            "source_crs": str(src_crs),
        },
    }


def save_geojson(geojson_dict: Dict[str, Any], output_path: Path) -> Path:
    """
    Saves a GeoJSON dict to file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_dict, f, indent=2)
    return output_path
