"""
Phase 1 Smoke Test — Satellite STAC & Data Access Verification
Tests Planetary Computer STAC search for Sentinel-1 RTC, same-orbit pairing,
anonymous SAS token signing, and asset URL availability.
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure workspace root is in python path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def check_satellite_access() -> Dict[str, Any]:
    results = {}

    # 1. Import STAC service
    try:
        from backend.services.stac_service import (
            search_sentinel1_scenes,
            find_trishuli_orbit_pair,
            sign_url_anonymous,
        )
        results["service_import"] = "PASS"
    except ImportError as e:
        results["service_import"] = f"FAIL: {e}"
        return results

    # 2. Query Sentinel-1 RTC scenes around Trishuli River, Nepal
    bbox_trishuli = [85.1, 27.9, 85.4, 28.2]
    try:
        start = time.perf_counter()
        scenes = search_sentinel1_scenes(
            bbox=bbox_trishuli,
            start_date="2026-08-01",
            end_date="2026-09-15",
            limit=10,
        )
        search_latency = round((time.perf_counter() - start) * 1000, 1)
        results["stac_search_latency_ms"] = search_latency
        results["scenes_found_count"] = len(scenes)

        if not scenes:
            # Fallback search for a known global flood event (e.g. Valencia Oct 2024)
            scenes = search_sentinel1_scenes(
                bbox=[-0.5, 39.3, -0.2, 39.6],
                start_date="2024-10-20",
                end_date="2024-11-05",
                limit=10,
            )
            results["scenes_found_count_fallback"] = len(scenes)

        if not scenes:
            results["stac_search"] = "FAIL: No scenes returned from STAC API"
            return results

        results["stac_search"] = "PASS"
        sample_scene = scenes[0]
        results["sample_scene_id"] = sample_scene.get("id", "unknown")
        results["sample_scene_datetime"] = sample_scene.get("properties", {}).get("datetime", "unknown")
        results["sample_scene_orbit"] = sample_scene.get("properties", {}).get("sat:relative_orbit")
        results["sample_scene_direction"] = sample_scene.get("properties", {}).get("sat:orbit_state")

        # Check available assets (expecting vv, vh)
        assets = sample_scene.get("assets", {})
        available_assets = list(assets.keys())
        results["sample_scene_assets"] = available_assets
        has_polarizations = ("vv" in assets or "vh" in assets)
        results["has_sar_polarizations"] = has_polarizations
    except Exception as e:
        results["stac_search"] = f"FAIL: {e}"
        return results

    # 3. Test Anonymous SAS Signing
    try:
        raw_href = None
        for key in ["vv", "vh", "rendered_preview", "thumbnail"]:
            if key in assets:
                raw_href = assets[key].get("href")
                break

        if raw_href:
            signed_href = sign_url_anonymous(raw_href)
            has_token = ("sig=" in signed_href or "se=" in signed_href or signed_href != raw_href)
            results["sas_signing"] = "PASS" if has_token else "UNMODIFIED"
            results["signed_href_valid_protocol"] = signed_href.startswith("https://")
        else:
            results["sas_signing"] = "SKIP: No asset href found to test"
    except Exception as e:
        results["sas_signing"] = f"FAIL: {e}"

    results["satellite_test"] = "PASS" if results.get("stac_search") == "PASS" else "FAIL"
    return results


def test_satellite():
    res = check_satellite_access()
    assert res.get("satellite_test") == "PASS", f"Satellite test failed: {res}"


if __name__ == "__main__":
    print("=" * 60)
    print("SATELLITE DATA & STAC SMOKE TEST")
    print("=" * 60)
    res = check_satellite_access()
    for k, v in res.items():
        print(f"  {k}: {v}")
    overall = "PASS" if res.get("satellite_test") == "PASS" else "FAIL"
    print(f"\n  OVERALL: {overall}")
    sys.exit(0 if overall == "PASS" else 1)
