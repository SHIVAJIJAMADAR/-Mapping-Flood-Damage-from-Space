"""
STAC Service for Microsoft Planetary Computer
Handles Sentinel-1 RTC and Copernicus DEM querying, same-orbit scene pair selection,
spatial coverage validation, and anonymous SAS token signing.
"""

import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from shapely.geometry import shape, box, Polygon

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_SAS_SIGN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


class STACError(Exception):
    """Base exception for STAC service errors."""
    pass


class NoCompatibleScenePairError(STACError):
    """Raised when no valid same-orbit pre/post satellite scene pair can be found."""
    pass


_SAS_CACHE: Dict[str, Tuple[str, float]] = {}


def sign_url_anonymous(href: str, max_retries: int = 4) -> str:
    """
    Signs an asset URL using Planetary Computer's anonymous SAS token endpoint.
    Caches signed tokens in-memory for 30 minutes to prevent redundant network round-trips
    and rate limiting. Retries with exponential backoff if transient errors occur.
    """
    if "blob.core.windows.net" not in href or "?" in href:
        return href

    import time
    now = time.time()
    if href in _SAS_CACHE:
        cached_signed_url, timestamp = _SAS_CACHE[href]
        # Valid if cached less than 30 minutes ago
        if now - timestamp < 1800:
            return cached_signed_url

    encoded_href = urllib.parse.quote(href, safe=":/?=&")
    url = f"{PC_SAS_SIGN_URL}?href={encoded_href}"
    req = urllib.request.Request(url, headers={"User-Agent": "FloodIntelligence/1.0"})

    last_err = None
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                signed = data.get("href", href)
                if "?" in signed:
                    _SAS_CACHE[href] = (signed, now)
                    return signed
                time.sleep(1.0 * (attempt + 1))
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))

    raise STACError(
        f"Failed to sign Planetary Computer asset URL after {max_retries} attempts: {href}. Error: {last_err}"
    )


def aoi_to_bbox_and_geom(aoi: Union[List[float], Dict[str, Any]]) -> Tuple[List[float], Polygon]:
    """
    Converts either a bbox list [minx, miny, maxx, maxy] or a GeoJSON dict into (bbox, Polygon).
    """
    if isinstance(aoi, (list, tuple)) and len(aoi) == 4:
        minx, miny, maxx, maxy = float(aoi[0]), float(aoi[1]), float(aoi[2]), float(aoi[3])
        return [minx, miny, maxx, maxy], box(minx, miny, maxx, maxy)
    elif isinstance(aoi, dict):
        geom = shape(aoi.get("geometry", aoi))
        bounds = geom.bounds
        return [bounds[0], bounds[1], bounds[2], bounds[3]], geom
    else:
        raise ValueError(f"Invalid AOI format: {aoi}. Expected bbox [minx, miny, maxx, maxy] or GeoJSON dict.")


def search_sentinel1_scenes(
    bbox: List[float],
    start_date: str,
    end_date: str,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    Queries sentinel-1-rtc collection on Planetary Computer for a given bounding box and date range.
    """
    # Ensure standard ISO datetime format
    start_iso = f"{start_date}T00:00:00Z" if "T" not in start_date else start_date
    end_iso = f"{end_date}T23:59:59Z" if "T" not in end_date else end_date

    payload = {
        "collections": ["sentinel-1-rtc"],
        "bbox": bbox,
        "datetime": f"{start_iso}/{end_iso}",
        "limit": limit,
    }
    req = urllib.request.Request(
        PC_STAC_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "FloodIntelligence/1.0"},
    )
    with urllib.request.urlopen(req, timeout=25) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("features", [])


def select_best_scene_pair(
    aoi: Union[List[float], Dict[str, Any]],
    event_date_str: str,
    search_window_days: int = 25,
    min_aoi_coverage_pct: float = 80.0,
    preferred_orbit: Optional[int] = None,
    preferred_orbit_state: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Discovers, validates, and ranks same-orbit Sentinel-1 RTC pre/post scene pairs.

    Parameters:
      aoi: [minx, miny, maxx, maxy] or GeoJSON geometry
      event_date_str: Flood event date in YYYY-MM-DD format
      search_window_days: Days before and after event_date to query
      min_aoi_coverage_pct: Minimum spatial overlap percentage (default 80%)
      preferred_orbit: Optional preferred relative orbit number (e.g. 19)
      preferred_orbit_state: Optional preferred orbit state (e.g. 'descending' or 'ascending')

    Returns a dict containing:
      - pre_scene: complete STAC metadata for pre-event observation
      - post_scene: complete STAC metadata for post-event observation
      - orbit: relative orbit number
      - orbit_state: ascending or descending
      - pre_datetime: ISO string
      - post_datetime: ISO string
      - pre_signed_urls: dict of signed asset URLs (vv, vh)
      - post_signed_urls: dict of signed asset URLs (vv, vh)
    """
    bbox, aoi_geom = aoi_to_bbox_and_geom(aoi)
    event_dt = datetime.strptime(event_date_str[:10], "%Y-%m-%d")
    start_date = (event_dt - timedelta(days=search_window_days)).strftime("%Y-%m-%d")
    end_date = (event_dt + timedelta(days=search_window_days)).strftime("%Y-%m-%d")

    scenes = search_sentinel1_scenes(bbox=bbox, start_date=start_date, end_date=end_date, limit=60)
    if not scenes:
        raise NoCompatibleScenePairError(
            f"No Sentinel-1 RTC scenes found in STAC catalogue for bbox={bbox} between {start_date} and {end_date}."
        )

    # Filter scenes by valid polarization (must have both vv and vh)
    valid_scenes = []
    for s in scenes:
        assets = s.get("assets", {})
        if "vv" in assets and "vh" in assets:
            valid_scenes.append(s)

    if not valid_scenes:
        raise NoCompatibleScenePairError("No Sentinel-1 RTC scenes contain both VV and VH polarization assets.")

    # Group by (relative_orbit, orbit_state)
    by_orbit: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for s in valid_scenes:
        props = s.get("properties", {})
        orbit = props.get("sat:relative_orbit")
        state = props.get("sat:orbit_state", "unknown")
        if orbit is not None:
            by_orbit.setdefault((orbit, state), []).append(s)

    # Candidate pairs evaluated across matching orbits
    candidates = []
    aoi_area = aoi_geom.area

    for (orbit, state), orbit_scenes in by_orbit.items():
        # Sort chronologically
        orbit_scenes.sort(key=lambda x: x["properties"].get("datetime", ""))

        pre_candidates = [
            s for s in orbit_scenes
            if datetime.fromisoformat(s["properties"]["datetime"].replace("Z", "+00:00")).date() < event_dt.date()
        ]
        post_candidates = [
            s for s in orbit_scenes
            if datetime.fromisoformat(s["properties"]["datetime"].replace("Z", "+00:00")).date() >= event_dt.date()
        ]

        if not pre_candidates or not post_candidates:
            continue

        # Test pair combinations starting from closest in time
        for pre_s in reversed(pre_candidates):  # closest pre first
            for post_s in post_candidates:     # closest post first
                pre_dt_val = datetime.fromisoformat(pre_s["properties"]["datetime"].replace("Z", "+00:00"))
                post_dt_val = datetime.fromisoformat(post_s["properties"]["datetime"].replace("Z", "+00:00"))

                # Check spatial coverage
                pre_geom = shape(pre_s.get("geometry", {}))
                post_geom = shape(post_s.get("geometry", {}))

                pre_overlap_pct = (pre_geom.intersection(aoi_geom).area / aoi_area) * 100.0 if aoi_area > 0 else 100.0
                post_overlap_pct = (post_geom.intersection(aoi_geom).area / aoi_area) * 100.0 if aoi_area > 0 else 100.0

                if pre_overlap_pct < min_aoi_coverage_pct or post_overlap_pct < min_aoi_coverage_pct:
                    continue

                # Total temporal distance metric (shorter is better)
                time_distance_days = (event_dt.date() - pre_dt_val.date()).days + (post_dt_val.date() - event_dt.date()).days

                candidates.append({
                    "score": time_distance_days,
                    "orbit": orbit,
                    "orbit_state": state,
                    "pre_scene": pre_s,
                    "post_scene": post_s,
                    "pre_datetime": pre_s["properties"]["datetime"],
                    "post_datetime": post_s["properties"]["datetime"],
                    "pre_coverage_pct": round(pre_overlap_pct, 1),
                    "post_coverage_pct": round(post_overlap_pct, 1),
                })

    if not candidates:
        raise NoCompatibleScenePairError(
            f"No same-orbit Sentinel-1 scene pair bracketing {event_date_str} with >={min_aoi_coverage_pct}% "
            f"coverage found for AOI={bbox}. Check window={search_window_days} days."
        )

    # Sort candidates by:
    # 1. preferred_orbit match (if requested)
    # 2. preferred_orbit_state match (if requested)
    # 3. minimal temporal spread (score)
    # 4. maximum pre and post spatial coverage
    def sort_key(c):
        orbit_match = 0 if (preferred_orbit is not None and c["orbit"] == preferred_orbit) else 1
        state_match = 0 if (preferred_orbit_state is not None and str(c["orbit_state"]).lower() == preferred_orbit_state.lower()) else 1
        return (orbit_match, state_match, c["score"], -c["pre_coverage_pct"], -c["post_coverage_pct"])

    candidates.sort(key=sort_key)
    best = candidates[0]

    # Generate signed URLs for assets
    pre_assets = best["pre_scene"]["assets"]
    post_assets = best["post_scene"]["assets"]

    best["pre_signed_urls"] = {
        "vv": sign_url_anonymous(pre_assets["vv"]["href"]),
        "vh": sign_url_anonymous(pre_assets["vh"]["href"]),
    }
    best["post_signed_urls"] = {
        "vv": sign_url_anonymous(post_assets["vv"]["href"]),
        "vh": sign_url_anonymous(post_assets["vh"]["href"]),
    }

    return best


def find_trishuli_orbit_pair(
    bbox: List[float] = [85.1, 27.9, 85.4, 28.2],
    event_date: str = "2026-08-26",
    preferred_orbit: Optional[int] = 19,
) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Finds the matching-orbit Sentinel-1 scene pair for the Trishuli case study.
    Defaults to the 2026-08-26 event date and relative orbit 19.
    Returns (pre_scene, post_scene) tuple.
    """
    try:
        pair_dict = select_best_scene_pair(
            aoi=bbox,
            event_date_str=event_date,
            preferred_orbit=preferred_orbit,
        )
        return pair_dict["pre_scene"], pair_dict["post_scene"]
    except NoCompatibleScenePairError:
        return None


def search_copernicus_dem(
    bbox: List[float],
) -> List[Dict[str, Any]]:
    """
    Queries cop-dem-glo-30 collection on Planetary Computer for a given bounding box.
    """
    payload = {
        "collections": ["cop-dem-glo-30"],
        "bbox": bbox,
        "limit": 6,
    }
    req = urllib.request.Request(
        PC_STAC_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "FloodIntelligence/1.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("features", [])


def get_signed_copernicus_dem_urls(bbox: List[float]) -> List[Dict[str, Any]]:
    """
    Returns signed URLs and spatial bounds for all Copernicus DEM tiles covering bbox.
    """
    items = search_copernicus_dem(bbox)
    if not items:
        raise STACError(f"No Copernicus GLO-30 DEM tiles found covering bbox={bbox}")

    results = []
    for item in items:
        assets = item.get("assets", {})
        data_asset = assets.get("data", {})
        raw_href = data_asset.get("href")
        if raw_href:
            signed_href = sign_url_anonymous(raw_href)
            results.append({
                "id": item.get("id"),
                "bbox": item.get("bbox"),
                "geometry": item.get("geometry"),
                "signed_href": signed_href,
            })
    return results
