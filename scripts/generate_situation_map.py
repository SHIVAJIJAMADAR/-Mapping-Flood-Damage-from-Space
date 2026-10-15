"""
Situation Map Generator for One-Page Situation Report
Renders Sentinel-1 SAR post-event imagery with SNUNet flood extent, classical change,
exposed roads, and settlement accessibility statuses.
"""

from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from PIL import Image, ImageDraw, ImageFont

def generate_situation_map():
    workspace = Path(__file__).resolve().parent.parent
    run_dir = workspace / "runs" / "flood_20261014_082338_45889d"
    sub_dir = workspace / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # 1. Master raster profile
    with rasterio.open(run_dir / "rasters" / "post_vv.tif") as src:
        post_vv = src.read(1)
        post_db = 10.0 * np.log10(np.clip(post_vv, 1e-5, None))
        master_shape = (src.height, src.width)
        master_transform = src.transform
        master_crs = src.crs

    H, W = master_shape

    # Normalize SAR to 8-bit grayscale
    norm_sar = np.clip((post_db - (-25.0)) / ((-3.0) - (-25.0)), 0.0, 1.0)
    base_gray = (norm_sar * 255.0).astype(np.uint8)
    base_img = Image.fromarray(base_gray, mode="L").convert("RGBA")

    # 2. Rasterize Classical Change (Amber)
    base_gdf = gpd.read_file(run_dir / "vectors" / "baseline_candidate_flood.geojson").to_crs(master_crs)
    base_mask = rasterize([(g, 1) for g in base_gdf.geometry], out_shape=master_shape, transform=master_transform, fill=0, dtype=np.uint8) == 1

    # 3. Rasterize SNUNet Flood Water (Electric Blue)
    snunet_gdf = gpd.read_file(run_dir / "vectors" / "snunet_flood_extent.geojson").to_crs(master_crs)
    snunet_mask = rasterize([(g, 1) for g in snunet_gdf.geometry], out_shape=master_shape, transform=master_transform, fill=0, dtype=np.uint8) == 1

    # 4. Rasterize Permanent River Water (Navy/Cyan)
    perm_path = run_dir / "vectors" / "snunet_permanent_water.geojson"
    if perm_path.is_file():
        perm_gdf = gpd.read_file(perm_path).to_crs(master_crs)
        perm_mask = rasterize([(g, 1) for g in perm_gdf.geometry], out_shape=master_shape, transform=master_transform, fill=0, dtype=np.uint8) == 1
    else:
        perm_mask = np.zeros(master_shape, dtype=bool)

    # 5. Composite Layers
    rgba_arr = np.zeros((H, W, 4), dtype=np.uint8)
    
    # Permanent water
    rgba_arr[perm_mask, 0] = 6
    rgba_arr[perm_mask, 1] = 182
    rgba_arr[perm_mask, 2] = 212
    rgba_arr[perm_mask, 3] = 180

    # Classical change
    rgba_arr[base_mask, 0] = 245
    rgba_arr[base_mask, 1] = 158
    rgba_arr[base_mask, 2] = 11
    rgba_arr[base_mask, 3] = 140

    # SNUNet Flood
    rgba_arr[snunet_mask, 0] = 0
    rgba_arr[snunet_mask, 1] = 210
    rgba_arr[snunet_mask, 2] = 255
    rgba_arr[snunet_mask, 3] = 230

    layer_img = Image.fromarray(rgba_arr, mode="RGBA")
    base_img.paste(layer_img, (0, 0), layer_img)

    # 6. Draw Vector Infrastructure Points and Lines
    draw = ImageDraw.Draw(base_img)

    # Potentially affected roads
    roads_gdf = gpd.read_file(run_dir / "derived" / "potentially_affected_roads.geojson").to_crs(master_crs)
    for geom in roads_gdf.geometry:
        if geom.geom_type == "LineString":
            pts = [~master_transform * (x, y) for x, y in geom.coords]
            draw.line(pts, fill=(239, 68, 68, 255), width=3)
        elif geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                pts = [~master_transform * (x, y) for x, y in part.coords]
                draw.line(pts, fill=(239, 68, 68, 255), width=3)

    # Settlements
    settle_gdf = gpd.read_file(run_dir / "derived" / "settlement_accessibility.geojson").to_crs(master_crs)
    for _, row in settle_gdf.iterrows():
        pt = row.geometry
        px, py = ~master_transform * (pt.x, pt.y)
        status = row.get("connectivity_status")
        name = row.get("name", "")

        if status == "cut_off":
            draw.ellipse([px-7, py-7, px+7, py+7], fill=(239, 68, 68, 255), outline=(255, 255, 255, 255), width=2)
            draw.text((px+9, py-6), f"{name} (CUT OFF)", fill=(239, 68, 68, 255))
        elif status == "detoured":
            draw.ellipse([px-5, py-5, px+5, py+5], fill=(245, 158, 11, 255), outline=(255, 255, 255, 255), width=1)
            draw.text((px+7, py-5), name, fill=(245, 158, 11, 255))
        else:
            draw.ellipse([px-3, py-3, px+3, py+3], fill=(16, 185, 129, 200), outline=(255, 255, 255, 200), width=1)

    # 7. Resize to clean report dimensions (1200 x 950)
    target_img = base_img.resize((1200, int(1200 * (H / W))), Image.Resampling.LANCZOS)
    out_path = sub_dir / "situation_map.png"
    target_img.convert("RGB").save(out_path, "PNG", quality=92)
    print(f"Generated situation map at {out_path}")

if __name__ == "__main__":
    generate_situation_map()
