"""
Phase 1 Smoke Test — Digital Elevation Model (DEM) Access Verification
Tests Planetary Computer STAC search for Copernicus DEM (cop-dem-glo-30),
anonymous SAS signing, and accessibility of elevation GeoTIFF assets.
"""
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, Any

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def check_dem_access() -> Dict[str, Any]:
    results = {}

    # 1. Import STAC service
    try:
        from backend.services.stac_service import search_copernicus_dem, sign_url_anonymous
        results["service_import"] = "PASS"
    except ImportError as e:
        results["service_import"] = f"FAIL: {e}"
        return results

    # 2. Query Copernicus DEM GLO-30 for Nepal Trishuli AOI
    bbox_trishuli = [85.1, 27.9, 85.4, 28.2]
    try:
        start = time.perf_counter()
        dem_scenes = search_copernicus_dem(bbox_trishuli)
        query_latency = round((time.perf_counter() - start) * 1000, 1)
        results["stac_dem_query_latency_ms"] = query_latency
        results["dem_scenes_found"] = len(dem_scenes)

        if not dem_scenes:
            results["dem_test"] = "FAIL: No DEM scenes found for Trishuli AOI"
            return results

        sample_dem = dem_scenes[0]
        results["sample_dem_id"] = sample_dem.get("id", "unknown")
        results["sample_dem_collection"] = sample_dem.get("collection", "unknown")

        assets = sample_dem.get("assets", {})
        results["available_dem_assets"] = list(assets.keys())

        # Check for 'data' asset
        data_asset = assets.get("data", {})
        raw_href = data_asset.get("href")
        if not raw_href:
            results["dem_test"] = "FAIL: No 'data' asset in DEM STAC item"
            return results

        results["raw_dem_href"] = raw_href[:80] + "..." if len(raw_href) > 80 else raw_href

        # 3. Sign DEM URL anonymously
        signed_href = sign_url_anonymous(raw_href)
        results["signed_dem_href_valid"] = signed_href.startswith("https://")
        results["sas_token_applied"] = ("sig=" in signed_href or "se=" in signed_href)

        # 4. HTTP Range Request to verify byte-level asset accessibility without downloading 70MB
        req = urllib.request.Request(
            signed_href,
            headers={
                "Range": "bytes=0-1023",
                "User-Agent": "FloodIntelligence/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            status = resp.status
            content_len = len(resp.read())
            results["http_status"] = status
            results["byte_range_returned_bytes"] = content_len

        if status in (200, 206) and content_len > 0:
            results["dem_test"] = "PASS"
        else:
            results["dem_test"] = f"FAIL: Unexpected HTTP status {status}"

    except Exception as e:
        results["dem_test"] = f"FAIL: {e}"

    return results


def test_dem():
    res = check_dem_access()
    assert res.get("dem_test") == "PASS", f"DEM test failed: {res}"


if __name__ == "__main__":
    print("=" * 60)
    print("DEM ACCESS SMOKE TEST")
    print("=" * 60)
    res = check_dem_access()
    for k, v in res.items():
        print(f"  {k}: {v}")
    overall = "PASS" if res.get("dem_test") == "PASS" else "FAIL"
    print(f"\n  OVERALL: {overall}")
    sys.exit(0 if overall == "PASS" else 1)
