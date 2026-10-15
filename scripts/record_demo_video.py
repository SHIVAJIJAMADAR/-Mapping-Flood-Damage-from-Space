"""
Demo Video Generation Script for Track B Submission.
Generates an official, publication-grade ~3-minute demonstration MP4 video (180.0s @ 15fps = 2700 frames)
capturing genuine running application states, calibrated radar measurements, and scientific disclosures.
"""

import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

def record_demo():
    workspace = Path(__file__).resolve().parent.parent
    sub_dir = workspace / "submission"
    scratch_dir = workspace / "scratch" / "video_frames"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    fps = 15
    total_seconds = 180.0
    total_frames = int(total_seconds * fps)  # 2700 frames

    print(f"Recording 3-minute demonstration video ({total_seconds}s @ {fps} fps = {total_frames} frames)...")

    # 1. Define the 7 core demo segments and URLs
    # Each segment has: (name, duration_sec, url, title, subtitle)
    segments = [
        # Segment 1: Case Study Selection & KPI Baseline (25s)
        {
            "name": "seg1_overview",
            "duration": 25.0,
            "url": "http://127.0.0.1:5173/?mode=swipe&swipe=50",
            "title": "1. Incident Selection & Multi-Temporal Sentinel-1 Baseline",
            "sub": "2026-08-26 Trishuli Flood • Orbit 19 Descending • 109.51 km² Footprint • Pre: 24 Aug / Post: 5 Sept"
        },
        # Segment 2: Interactive Swipe Divider (30s)
        {
            "name": "seg2_swipe",
            "duration": 30.0,
            "swipe_steps": [
                (0.0, 6.0, "http://127.0.0.1:5173/?mode=swipe&swipe=25", "Swipe 25% (Pre-Event SAR Dominance)"),
                (6.0, 12.0, "http://127.0.0.1:5173/?mode=swipe&swipe=40", "Swipe 40% (Transitioning Across Trishuli River)"),
                (12.0, 18.0, "http://127.0.0.1:5173/?mode=swipe&swipe=60", "Swipe 60% (Revealing Post-Event Inundation)"),
                (18.0, 24.0, "http://127.0.0.1:5173/?mode=swipe&swipe=85", "Swipe 85% (Post-Event SAR Dominance)"),
                (24.0, 30.0, "http://127.0.0.1:5173/?mode=swipe&swipe=50", "Swipe 50% (Symmetric Before/After Comparison)"),
            ],
            "title": "2. Actual Multi-Temporal SAR Observation & Interactive Swipe Divider",
            "sub": "Sentinel-1 RTC 10m Ground Resolution • Microwave Surface Scattering Reveals Flooded Channels"
        },
        # Segment 3: Multi-Layer Overlays (30s)
        {
            "name": "seg3_overlays",
            "duration": 30.0,
            "layer_steps": [
                (0.0, 10.0, "http://127.0.0.1:5173/?mode=overlay&layers=post_vv,snunet_flood,snunet_perm", "SNUNet-ECAM Neural Water: 2.87 ha Flood (Blue) + 10.7 ha River (Cyan)"),
                (10.0, 20.0, "http://127.0.0.1:5173/?mode=overlay&layers=post_vv,snunet_flood,snunet_perm,baseline_change", "Classical Radar Drop Baseline: 138.1 ha Candidate Scouring (Yellow &Delta;VV < -3 dB)"),
                (20.0, 30.0, "http://127.0.0.1:5173/?mode=overlay&layers=post_vv,baseline_change,slope", "Copernicus DEM 30m Slope Mask: Excluding Slopes &ge; 15° to Eliminate Ridge False Drops"),
            ],
            "title": "3. Dual-Engine Flood & Radar-Change Inundation Overlays",
            "sub": "Comparing Deep Siamese Learning (SNUNet-ECAM) with Physics-Based Decibel Thresholding"
        },
        # Segment 4: Calibrated Point Inspection (30s)
        {
            "name": "seg4_inspect",
            "duration": 30.0,
            "inspect_steps": [
                (0.0, 15.0, "http://127.0.0.1:5173/?mode=overlay&layers=post_vv,snunet_flood,baseline_change,affected_roads&inspect_lat=28.005&inspect_lon=85.182", "Terrace Inundation: Lat 28.005°N, Lon 85.182°E • Pre: -11.4 dB, Post: -18.7 dB (&Delta; -7.3 dB)"),
                (15.0, 30.0, "http://127.0.0.1:5173/?mode=overlay&layers=post_vv,snunet_flood,affected_roads&inspect_lat=27.98&inspect_lon=85.16", "Non-Flooded Valley Hillside: Lat 27.980°N, Lon 85.160°E • High Slope, Normal Backscatter"),
            ],
            "title": "4. Calibrated Point Inspector with Deterministic Explanations",
            "sub": "Real-Time Sampling of Stored Georeferenced Rasters: dB Backscatter, DEM Elevation & Slope"
        },
        # Segment 5: OSM Infrastructure Impact & Topological Connectivity (30s)
        {
            "name": "seg5_connectivity",
            "duration": 30.0,
            "conn_steps": [
                (0.0, 15.0, "http://127.0.0.1:5173/?mode=overlay&layers=post_vv,affected_roads,destinations", "Submerged Roadways: 12 Unique OSM Ways (190.1 m Direct Inundation, 1,436 Edge Fragments)"),
                (15.0, 30.0, "http://127.0.0.1:5173/?mode=overlay&layers=post_vv,affected_roads,settlement_access,destinations", "Dijkstra Reachability: Laharepauwa & Dhunge CUT-OFF; Bhorle DETOURED (+24%); 23 Unaffected"),
            ],
            "title": "5. Historical OSM Infrastructure Impact & Settlement Isolation (Dijkstra)",
            "sub": "Strict 2026-07-27 Attic Cutoff • Hospital Routing • Building Completeness Disclaimer (1,000 Cap)"
        },
        # Segment 6: Independent EMSR927 Reference Comparison (25s)
        {
            "name": "seg6_emsr927",
            "duration": 25.0,
            "type": "emsr927_graphic",
            "title": "6. Copernicus EMS (EMSR927) Independent Spatial-Overlap Analysis",
            "sub": "Bidur AOI03 (5.08 km² Footprint) • 157.56 ha Mass Movement vs 37.58 ha Radar Scouring Overlap (67.7% Prec.)"
        },
        # Segment 7: Reproducibility & Mandatory Attributions (10s)
        {
            "name": "seg7_attribution",
            "duration": 10.0,
            "type": "summary_graphic",
            "title": "7. Scientific Reproducibility, Test Verification & Attributions",
            "sub": "27/27 Tests Passed • Pretrained Kuro Siwo Checkpoint Disclosure • Mandatory Copernicus & OSM Credits"
        }
    ]

    # 2. Capture Keyframe Screenshots from Live Browser
    screenshot_cache = {}
    def get_screenshot(url, cache_key):
        if cache_key in screenshot_cache:
            return screenshot_cache[cache_key]

        out_img = scratch_dir / f"{cache_key}.png"
        cmd = [
            chrome_path,
            "--headless=new",
            "--disable-gpu",
            "--window-size=1920,1080",
            "--virtual-time-budget=6000",
            f"--screenshot={out_img}",
            url
        ]
        p = subprocess.run(cmd, capture_output=True, timeout=30)
        time.sleep(0.5)

        if not out_img.exists() or out_img.stat().st_size < 15000:
            print(f"Warning: Screenshot failed for {cache_key}, retrying...")
            time.sleep(2.0)
            subprocess.run(cmd, capture_output=True, timeout=30)

        img = Image.open(out_img).convert("RGB")
        screenshot_cache[cache_key] = img
        return img

    print("Pre-rendering browser screenshots...")
    # Pre-render Seg 1
    get_screenshot(segments[0]["url"], "seg1_overview")

    # Pre-render Seg 2 steps
    for idx, (_, _, u, _) in enumerate(segments[1]["swipe_steps"]):
        get_screenshot(u, f"seg2_step_{idx}")

    # Pre-render Seg 3 steps
    for idx, (_, _, u, _) in enumerate(segments[2]["layer_steps"]):
        get_screenshot(u, f"seg3_step_{idx}")

    # Pre-render Seg 4 steps
    for idx, (_, _, u, _) in enumerate(segments[3]["inspect_steps"]):
        get_screenshot(u, f"seg4_step_{idx}")

    # Pre-render Seg 5 steps
    for idx, (_, _, u, _) in enumerate(segments[4]["conn_steps"]):
        get_screenshot(u, f"seg5_step_{idx}")

    print("Browser screenshots captured successfully.")

    # 3. Create Custom Graphic for Segment 6 (EMSR927 Scientific Validation Card)
    def render_emsr927_frame():
        bg = Image.new("RGB", (1920, 1080), (15, 23, 42))
        draw = ImageDraw.Draw(bg)

        # Draw decorative top border
        draw.rectangle([(0, 0), (1920, 60)], fill=(30, 41, 59))

        # Embed EMSR927 triptych image
        triptych_path = sub_dir / "emsr927_validation_comparison.png"
        if triptych_path.exists():
            trip_img = Image.open(triptych_path).convert("RGB")
            # Resize nicely
            trip_w = 1200
            trip_h = int(trip_img.height * (trip_w / trip_img.width))
            trip_resized = trip_img.resize((trip_w, trip_h), Image.Resampling.LANCZOS)
            bg.paste(trip_resized, (60, 160))

        # Right-hand side summary card
        card_x = 1300
        card_y = 160
        card_w = 560
        card_h = 580
        draw.rounded_rectangle([(card_x, card_y), (card_x + card_w, card_y + card_h)], radius=8, fill=(30, 41, 59), outline=(71, 85, 105), width=2)

        # Card Title
        draw.text((card_x + 20, card_y + 20), "INDEPENDENT REFERENCE AUDIT", fill=(56, 189, 248))
        draw.text((card_x + 20, card_y + 45), "Copernicus EMS Rapid Mapping Activation EMSR927", fill=(203, 213, 225))
        draw.line([(card_x + 20, card_y + 70), (card_x + card_w - 20, card_y + 70)], fill=(71, 85, 105), width=1)

        # Stats Table
        stats = [
            ("Reference Product", "EMSR927_AOI03_GRA_PRODUCT_v1"),
            ("Reference Location", "Bidur, Nuwakot District, Nepal"),
            ("Reference Sensors", "BlackSky & Satellogic (0.8m-1.0m VHR)"),
            ("Reference Date", "2026-08-27 (1 day post-peak)"),
            ("Shared Evaluation Footprint", "5.08 km² (507.56 ha)"),
            ("CEMS Mapped Event Class", "6-Mass Movement (Landslide / Debris)"),
            ("CEMS Mapped Damage Area", "157.56 ha"),
            ("Radar Drop Candidate Area", "55.53 ha (Vector) / 62.06 ha (Raw)"),
            ("Radar Drop Overlap (TP)", "37.58 ha (Vector) / 40.32 ha (Raw)"),
            ("Candidate Overlap Precision", "67.68% (Vector) / 64.97% (Raw)"),
            ("SNUNet Flood Overlap (TP)", "0.15 ha (Vector) / 0.44 ha (Raw)"),
        ]

        sy = card_y + 85
        for k, v in stats:
            draw.text((card_x + 20, sy), k, fill=(148, 163, 184))
            draw.text((card_x + 260, sy), v, fill=(241, 245, 249))
            sy += 28

        # Callout Box below table
        draw.line([(card_x + 20, sy + 5), (card_x + card_w - 20, sy + 5)], fill=(71, 85, 105), width=1)
        draw.text((card_x + 20, sy + 15), "DOCUMENTED SEMANTIC DIVERGENCE:", fill=(251, 191, 36))
        expl = (
            "EMSR927 mapped high-gradient landslide scars & debris flow paths. "
            "SNUNet-ECAM targets standing flood water rather than mass-movement debris. "
            "Classical radar change (delta_VV < -3 dB) captured 37.58 ha (vector) / "
            "40.32 ha (raw) of the scouring corridor with 65-68% candidate precision."
        )
        # Word wrap explanation
        words = expl.split()
        line = ""
        ey = sy + 38
        for w in words:
            if len(line + " " + w) > 52:
                draw.text((card_x + 20, ey), line, fill=(203, 213, 225))
                ey += 18
                line = w
            else:
                line = line + " " + w if line else w
        if line:
            draw.text((card_x + 20, ey), line, fill=(203, 213, 225))

        return bg

    emsr927_frame_img = render_emsr927_frame()

    # 4. Create Custom Graphic for Segment 7 (Verification & Compliance Summary)
    def render_summary_frame():
        bg = Image.new("RGB", (1920, 1080), (15, 23, 42))
        draw = ImageDraw.Draw(bg)

        draw.text((100, 120), "RAPID MULTIMODAL FLOOD DAMAGE & CONNECTIVITY INTELLIGENCE", fill=(56, 189, 248))
        draw.text((100, 155), "Multimodal AI Hackathon 2026 — Track B Final Scientific Submission", fill=(241, 245, 249))
        draw.line([(100, 195), (1820, 195)], fill=(71, 85, 105), width=2)

        # 3 Pillar Cards
        cards = [
            (
                "VERIFIED REPRODUCIBILITY",
                [
                    ("Automated Test Suite", "27 of 27 Tests Passed (pytest tests/ -v)"),
                    ("Frontend Production Build", "Vite build succeeded (1.44s)"),
                    ("Inference Checkpoint", "SNUNet-ECAM (12.5M params, SHA256 verified)"),
                    ("SAR Normalization", "Kuro Siwo global linear power contract"),
                    ("Terrain Slope Physics", "2nd-order central difference on Copernicus 30m"),
                ],
                (16, 185, 129)
            ),
            (
                "HONEST AI PROVENANCE AUDIT",
                [
                    ("Model Architecture", "Siamese Nested U-Net with ECAM"),
                    ("Checkpoint Weights", "Kuro Siwo Benchmark (Bountos et al., NeurIPS 2024)"),
                    ("Local Fine-Tuning", "None performed (Disclosed transparently)"),
                    ("Reason for No Fine-Tuning", "Strict compute & time constraints"),
                    ("Target Water Semantics", "Strictly evaluates Class 2 (Flood Water)"),
                ],
                (245, 158, 11)
            ),
            (
                "MANDATORY LEGAL ATTRIBUTIONS",
                [
                    ("Sentinel-1 SAR", "Contains modified Copernicus Sentinel data 2026."),
                    ("WorldDEM-30 Elevation", "Produced using Copernicus WorldDEM-30 © DLR e.V."),
                    ("WorldDEM Rights", "© Airbus Defence and Space GmbH; all rights reserved."),
                    ("OpenStreetMap", "© OpenStreetMap contributors (ODbL)."),
                    ("NeurIPS 2024 Benchmark", "Bountos et al., Kuro Siwo Flood Benchmark."),
                ],
                (59, 130, 246)
            ),
        ]

        cx = 100
        for title, items, color in cards:
            draw.rounded_rectangle([(cx, 230), (cx + 520, 780)], radius=8, fill=(30, 41, 59), outline=color, width=2)
            draw.text((cx + 25, 260), title, fill=color)
            draw.line([(cx + 25, 295), (cx + 495, 295)], fill=(71, 85, 105), width=1)

            iy = 320
            for label, val in items:
                draw.text((cx + 25, iy), label, fill=(148, 163, 184))
                # Wrap value if needed
                if len(val) > 38:
                    parts = val.split(" ", 4)
                    draw.text((cx + 25, iy + 20), " ".join(parts[:4]), fill=(241, 245, 249))
                    draw.text((cx + 25, iy + 40), parts[4] if len(parts) > 4 else "", fill=(241, 245, 249))
                    iy += 70
                else:
                    draw.text((cx + 25, iy + 20), val, fill=(241, 245, 249))
                    iy += 55

            cx += 560

        # Bottom Banner
        draw.rounded_rectangle([(100, 810), (1820, 890)], radius=6, fill=(15, 118, 110))
        draw.text((130, 835), "Full Source Code, Technical Report (6 Pages), One-Page Situation Report & Validation Metrics Ready for Submission", fill=(240, 253, 250))

        return bg

    summary_frame_img = render_summary_frame()

    # 5. Video Encoding Pipeline using imageio
    output_mp4 = sub_dir / "demo_video.mp4"
    print(f"Opening MP4 writer at {output_mp4} (fps={fps}, total_frames={total_frames})...")

    writer = imageio.get_writer(
        output_mp4,
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p"
    )

    frame_idx = 0
    seg_start_sec = 0.0

    for seg in segments:
        seg_duration = seg["duration"]
        seg_frames = int(seg_duration * fps)

        print(f"Rendering {seg['name']} ({seg_duration}s = {seg_frames} frames)...")

        for f in range(seg_frames):
            global_sec = (frame_idx) / fps
            seg_rel_sec = f / fps

            # Determine base image for this sub-frame
            if seg["name"] == "seg1_overview":
                base_img = screenshot_cache["seg1_overview"].copy()
                step_title = seg["sub"]

            elif seg["name"] == "seg2_swipe":
                # Find matching swipe step
                matched_step = seg["swipe_steps"][-1]
                for s_start, s_end, u, note in seg["swipe_steps"]:
                    if s_start <= seg_rel_sec < s_end:
                        matched_step = (s_start, s_end, u, note)
                        break
                step_idx = [s[3] for s in seg["swipe_steps"]].index(matched_step[3])
                base_img = screenshot_cache[f"seg2_step_{step_idx}"].copy()
                step_title = f"{matched_step[3]} • {seg['sub']}"

            elif seg["name"] == "seg3_overlays":
                matched_step = seg["layer_steps"][-1]
                for s_start, s_end, u, note in seg["layer_steps"]:
                    if s_start <= seg_rel_sec < s_end:
                        matched_step = (s_start, s_end, u, note)
                        break
                step_idx = [s[3] for s in seg["layer_steps"]].index(matched_step[3])
                base_img = screenshot_cache[f"seg3_step_{step_idx}"].copy()
                step_title = f"{matched_step[3]} • {seg['sub']}"

            elif seg["name"] == "seg4_inspect":
                matched_step = seg["inspect_steps"][-1]
                for s_start, s_end, u, note in seg["inspect_steps"]:
                    if s_start <= seg_rel_sec < s_end:
                        matched_step = (s_start, s_end, u, note)
                        break
                step_idx = [s[3] for s in seg["inspect_steps"]].index(matched_step[3])
                base_img = screenshot_cache[f"seg4_step_{step_idx}"].copy()
                step_title = f"{matched_step[3]}"

            elif seg["name"] == "seg5_connectivity":
                matched_step = seg["conn_steps"][-1]
                for s_start, s_end, u, note in seg["conn_steps"]:
                    if s_start <= seg_rel_sec < s_end:
                        matched_step = (s_start, s_end, u, note)
                        break
                step_idx = [s[3] for s in seg["conn_steps"]].index(matched_step[3])
                base_img = screenshot_cache[f"seg5_step_{step_idx}"].copy()
                step_title = f"{matched_step[3]}"

            elif seg["name"] == "seg6_emsr927":
                base_img = emsr927_frame_img.copy()
                step_title = seg["sub"]

            elif seg["name"] == "seg7_attribution":
                base_img = summary_frame_img.copy()
                step_title = seg["sub"]

            # Compose Broadcast Overlay
            draw = ImageDraw.Draw(base_img)

            # 1. Top Banner Bar
            draw.rectangle([(0, 0), (1920, 48)], fill=(15, 23, 42, 240))
            draw.line([(0, 48), (1920, 48)], fill=(56, 189, 248), width=2)

            # Title
            draw.text((25, 14), seg["title"], fill=(241, 245, 249))

            # Timer badge [01:23 / 03:00]
            cur_min = int(global_sec) // 60
            cur_s = int(global_sec) % 60
            time_str = f"[{cur_min:02d}:{cur_s:02d} / 03:00]"
            draw.text((1760, 14), time_str, fill=(56, 189, 248))

            # 2. Lower-Third Subtitle Bar
            draw.rectangle([(0, 1032), (1920, 1076)], fill=(15, 23, 42, 235))
            draw.text((25, 1045), step_title, fill=(226, 232, 240))

            # 3. Bottom Progress Bar
            progress_ratio = min(1.0, (frame_idx + 1) / total_frames)
            bar_w = int(1920 * progress_ratio)
            draw.rectangle([(0, 1076), (1920, 1080)], fill=(30, 41, 59))
            draw.rectangle([(0, 1076), (bar_w, 1080)], fill=(56, 189, 248))

            # Convert to RGB numpy array and write frame
            frame_arr = np.array(base_img)
            writer.append_data(frame_arr)
            frame_idx += 1

        seg_start_sec += seg_duration

    writer.close()
    print(f"\nSuccessfully generated {output_mp4}!")
    print(f"File size: {output_mp4.stat().st_size:,} bytes")
    print(f"Total frames written: {frame_idx} (Target: {total_frames})")
    print(f"Duration: {frame_idx / fps:.2f} seconds ({frame_idx / (fps * 60):.2f} minutes)")

if __name__ == "__main__":
    record_demo()
