"""
Generates an animated multi-frame demonstration walkthrough (demo_walkthrough.webp)
illustrating the dynamic swipe comparison, multi-layer overlay, point inspector drawer,
and EMSR927 independent validation.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def generate_walkthrough():
    workspace = Path(__file__).resolve().parent.parent
    sub_dir = workspace / "submission"
    run_dir = workspace / "runs" / "flood_20261014_082338_45889d"

    # Load base images
    sit_map_path = sub_dir / "situation_map.png"
    val_map_path = sub_dir / "emsr927_validation_comparison.png"

    if not sit_map_path.is_file() or not val_map_path.is_file():
        print("Required images not found.")
        return

    sit_img = Image.open(sit_map_path).convert("RGBA").resize((1200, 750))
    val_img = Image.open(val_map_path).convert("RGBA").resize((1200, 750))

    # Also load pre and post SAR to create realistic swipe animation frames
    import rasterio
    with rasterio.open(run_dir / "rasters" / "pre_vv.tif") as s_pre:
        pre_db = 10.0 * np.log10(np.clip(s_pre.read(1), 1e-5, None))
    with rasterio.open(run_dir / "rasters" / "post_vv.tif") as s_post:
        post_db = 10.0 * np.log10(np.clip(s_post.read(1), 1e-5, None))

    pre_norm = (np.clip((pre_db - (-25.0)) / 22.0, 0.0, 1.0) * 255.0).astype(np.uint8)
    post_norm = (np.clip((post_db - (-25.0)) / 22.0, 0.0, 1.0) * 255.0).astype(np.uint8)

    pre_pil = Image.fromarray(pre_norm, mode="L").convert("RGBA").resize((1200, 750))
    post_pil = Image.fromarray(post_norm, mode="L").convert("RGBA").resize((1200, 750))

    frames = []

    # Frame generator helper
    def make_dashboard_frame(title_extra: str, map_content: Image.Image, inspector_text: str = None):
        frame = Image.new("RGBA", (1280, 850), (15, 23, 42, 255))
        draw = ImageDraw.Draw(frame)

        # Header bar
        draw.rectangle([0, 0, 1280, 55], fill=(30, 41, 59, 255))
        draw.text((20, 15), "FLOOD INTELLIGENCE PLATFORM — TRACK B: MAPPING FLOOD DAMAGE FROM SPACE", fill=(241, 245, 249, 255))
        draw.text((880, 15), "Sentinel-1 Orbit 19 | Trishuli Corridor, Nepal", fill=(56, 189, 248, 255))

        # Sub-bar
        draw.rectangle([0, 55, 1280, 85], fill=(24, 34, 50, 255))
        draw.text((20, 62), f"MODE: {title_extra}", fill=(245, 158, 11, 255))

        # Paste map
        frame.paste(map_content, (40, 95))

        # Optional Inspector Drawer
        if inspector_text:
            draw.rectangle([880, 95, 1240, 845], fill=(30, 41, 59, 240), outline=(56, 189, 248, 255), width=2)
            draw.text((900, 115), "PIXEL INSPECTION", fill=(56, 189, 248, 255))
            y = 150
            for line in inspector_text.split("\n"):
                draw.text((900, y), line, fill=(241, 245, 249, 255))
                y += 24

        return frame

    # 1. Title Sequence (2 frames)
    f_intro = make_dashboard_frame("Operational Overview (August 2026 GLOF Event)", sit_img)
    for _ in range(3):
        frames.append(f_intro)

    # 2. Swipe Motion Sequence (Swipe divider moving from 15% to 85% and back)
    W_m, H_m = 1200, 750
    for split_pct in [0.2, 0.35, 0.5, 0.65, 0.8, 0.65, 0.5]:
        split_x = int(W_m * split_pct)
        swipe_img = pre_pil.copy()
        post_crop = post_pil.crop((split_x, 0, W_m, H_m))
        swipe_img.paste(post_crop, (split_x, 0))

        # Draw divider line and badges
        s_draw = ImageDraw.Draw(swipe_img)
        s_draw.line([(split_x, 0), (split_x, H_m)], fill=(0, 210, 255, 255), width=3)
        s_draw.rectangle([20, 20, 240, 50], fill=(15, 23, 42, 220), outline=(56, 189, 248, 255))
        s_draw.text((30, 28), "PRE-EVENT: 2026-08-24", fill=(56, 189, 248, 255))
        s_draw.rectangle([W_m - 260, 20, W_m - 20, 50], fill=(15, 23, 42, 220), outline=(245, 158, 11, 255))
        s_draw.text((W_m - 250, 28), "POST-EVENT: 2026-09-05", fill=(245, 158, 11, 255))

        f_swipe = make_dashboard_frame(f"Swipe Comparison ({int(split_pct*100)}% Divider)", swipe_img)
        frames.append(f_swipe)

    # 3. Multi-Layer Overlay & Point Inspector Sequence
    insp_text = (
        "Lon: 85.20000 | Lat: 28.00000\n"
        "Pre VV: -3.85 dB\n"
        "Post VV: -6.71 dB\n"
        "Delta VV: -2.86 dB\n"
        "Slope: 27.5 deg | Elev: 1215m\n"
        "SNUNet Class: 0 (No Water)\n"
        "SNUNet Prob: 0.0%\n"
        "Classical: Not Met\n\n"
        "REASONING:\n"
        "A backscatter decrease of\n"
        "2.86 dB is measured. The\n"
        "classical drop (< -3 dB) is\n"
        "not met and slope (27.5 deg)\n"
        "exceeds 15 deg filter.\n"
        "SNUNet classifies as dry\n"
        "land (0% flood probability).\n\n"
        "INFRASTRUCTURE:\n"
        "Exposed Road: 49m away\n"
        "Laharepauwa: CUT-OFF\n"
        "Dhunge: CUT-OFF"
    )
    f_inspect = make_dashboard_frame("Multi-Layer Overlay & Clickable Point Inspector", sit_img, insp_text)
    for _ in range(4):
        frames.append(f_inspect)

    # 4. EMSR927 Validation Benchmark Sequence
    f_val = make_dashboard_frame("EMSR927 Independent Reference Benchmark (157.6 ha Debris Flow)", val_img)
    for _ in range(4):
        frames.append(f_val)

    # Save as animated WebP
    out_webp = sub_dir / "demo_walkthrough.webp"
    frames[0].save(
        out_webp,
        "WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=650,
        loop=0,
        quality=90
    )
    print(f"Generated demo walkthrough animation at {out_webp} ({len(frames)} frames)")

if __name__ == "__main__":
    generate_walkthrough()
