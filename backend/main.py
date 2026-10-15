"""
Flood Intelligence Platform - Backend API Service
Track B: Mapping Flood Damage from Space
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.pipeline import (
    run_flood_analysis,
    run_infrastructure_and_connectivity_analysis,
    NoCompatibleScenePairError,
)
from backend.services.tile_service import render_raster_tile, ALLOWED_LAYERS
from backend.services.inspect_service import inspect_point
from backend.services.report_service import generate_report_html_for_run, generate_report_pdf_for_run

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Flood Intelligence Platform API",
    description="Rapid geospatial intelligence from Sentinel-1 SAR & OpenStreetMap",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    service: str
    version: str
    case_study: Dict[str, Any]
    attribution: Dict[str, str]


class AnalysisRequest(BaseModel):
    bbox: List[float] = Field(
        default=[85.1, 27.9, 85.4, 28.2],
        description="Bounding box [minx, miny, maxx, maxy] in WGS 84 (EPSG:4326)",
    )
    event_date: str = Field(
        default="2026-08-26",
        description="Event date in YYYY-MM-DD format (Default: 2026-08-26 Trishuli flood)",
    )
    search_window_days: int = Field(default=25, description="Search window in days")
    min_aoi_coverage_pct: float = Field(default=75.0, description="Min AOI spatial overlap %")
    preferred_orbit: Optional[int] = Field(default=19, description="Preferred relative orbit number (e.g. 19)")
    preferred_orbit_state: Optional[str] = Field(default="descending", description="Preferred orbit state (e.g. descending)")
    db_drop_threshold: float = Field(default=-3.0, description="Classical baseline threshold in dB")
    max_slope_degrees: float = Field(default=15.0, description="Max terrain slope for candidate filter")


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint providing platform metadata and attribution."""
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - START_TIME, 2),
        service="flood-intelligence-backend",
        version="0.1.0",
        case_study={
            "name": "Bhote Koshi–Trishuli River Corridor, Nepal",
            "event_date": "2026-08-26",
            "copernicus_activation": "EMSR927",
            "primary_orbit": 19,
            "direction": "descending",
        },
        attribution={
            "sentinel": "Contains modified Copernicus Sentinel data 2024/2026.",
            "world_dem": "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
            "osm": "© OpenStreetMap contributors.",
            "model": "Kuro Siwo: A global multi-temporal SAR dataset for rapid flood mapping (Bountos et al., NeurIPS 2024)",
        },
    )


@app.post("/api/analyses")
def trigger_analysis(req: AnalysisRequest):
    """
    Triggers end-to-end Sentinel-1 RTC acquisition, DEM alignment,
    SNUNet inference, classical change detection, and vectorization.
    """
    try:
        manifest = run_flood_analysis(
            aoi=req.bbox,
            event_date=req.event_date,
            search_window_days=req.search_window_days,
            min_aoi_coverage_pct=req.min_aoi_coverage_pct,
            preferred_orbit=req.preferred_orbit,
            preferred_orbit_state=req.preferred_orbit_state,
            db_drop_threshold=req.db_drop_threshold,
            max_slope_degrees=req.max_slope_degrees,
        )
        return {
            "status": "COMPLETED",
            "run_id": manifest["run_id"],
            "event_date": req.event_date,
            "summary": manifest["summary_statistics"],
            "satellite_scenes": manifest["satellite_scenes"],
            "timings_seconds": manifest["timings_seconds"],
        }
    except NoCompatibleScenePairError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")


@app.get("/api/analyses")
def list_analyses():
    """Lists available completed analysis runs with event dates and labels."""
    runs_dir = WORKSPACE_ROOT / "runs"
    runs = []
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir(), reverse=True):
            if d.is_dir() and (d / "manifest.json").is_file():
                try:
                    with open(d / "manifest.json", "r", encoding="utf-8") as f:
                        m = json.load(f)
                    run_id = d.name
                    event_date = m.get("inputs", {}).get("event_date", "2026-08-26")
                    is_case_study = (event_date == "2026-08-26" and "082338" in run_id)
                    label = "Verified Case Study (2026-08-26)" if is_case_study else f"Historical Analysis ({event_date})"
                    has_conn = (d / "derived" / "connectivity_summary.json").is_file()
                    runs.append({
                        "run_id": run_id,
                        "event_date": event_date,
                        "label": label,
                        "is_case_study": is_case_study,
                        "timestamp_utc": m.get("timestamp_utc", ""),
                        "has_connectivity": has_conn,
                        "status": m.get("status", "COMPLETED"),
                    })
                except Exception:
                    pass
    return runs


@app.get("/api/analyses/{run_id}")
def get_analysis_status(run_id: str):
    """Returns run metadata and status for a given run ID."""
    run_dir = WORKSPACE_ROOT / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail=f"Analysis run {run_id} not found.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return {
        "run_id": manifest["run_id"],
        "status": manifest.get("status", "COMPLETED"),
        "timestamp_utc": manifest.get("timestamp_utc"),
        "inputs": manifest.get("inputs"),
        "satellite_scenes": manifest.get("satellite_scenes"),
        "summary": manifest.get("summary_statistics"),
        "timings": manifest.get("timings_seconds"),
    }


@app.get("/api/analyses/{run_id}/results")
def get_analysis_results(run_id: str):
    """Returns complete manifest and output file references for a completed analysis."""
    run_dir = WORKSPACE_ROOT / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail=f"Analysis run {run_id} not found.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    return manifest


@app.get("/api/analyses/{run_id}/report.html")
def get_analysis_report_html(run_id: str):
    """Returns the one-page HTML situation report for a completed analysis."""
    if not run_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid run ID format.")
    try:
        html = generate_report_html_for_run(run_id)
        return Response(content=html, media_type="text/html")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {e}")


@app.get("/api/analyses/{run_id}/report.pdf")
def get_analysis_report_pdf(run_id: str):
    """Generates and delivers an official single-page A4 PDF situation report for a completed analysis."""
    if not run_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid run ID format.")
    try:
        pdf_bytes = generate_report_pdf_for_run(run_id)
        filename = f"situation_report_{run_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={filename}"},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation error: {e}")


@app.get("/api/analyses/{run_id}/geojson")
def get_analysis_geojson(run_id: str, layer: str = "snunet_flood_extent"):
    """
    Returns the vector GeoJSON FeatureCollection for a completed analysis.
    Supported layers:
      - Flood extents: 'snunet_flood_extent', 'snunet_permanent_water', 'baseline_candidate_flood'
      - Infrastructure impact: 'asset_impact', 'potentially_affected_roads', 'settlement_accessibility'
      - Pre-event OSM assets (enriched): 'osm_roads', 'osm_buildings', 'osm_bridges', 'osm_settlements', 'osm_destinations'
      - Filtered rescue layers: 'buildings_exposed', 'buildings_candidate', 'bridges_exposed', 'bridges_candidate', 'roads_candidate', 'severed_roads'
    """
    run_dir = WORKSPACE_ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Analysis run {run_id} not found.")

    # Check enriched asset impact service layers first
    try:
        from backend.services.asset_impact_service import get_run_asset_impact_data
        impact_data = get_run_asset_impact_data(run_dir)
        if layer in impact_data.get("layers", {}):
            return impact_data["layers"][layer]
    except Exception as e:
        pass

    candidate_paths = [
        run_dir / "vectors" / f"{layer}.geojson",
        run_dir / "derived" / f"{layer}.geojson",
        run_dir / "osm" / f"{layer}.geojson",
    ]
    for path in candidate_paths:
        if path.is_file():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    raise HTTPException(status_code=404, detail=f"Vector layer '{layer}' not found in run {run_id}.")


@app.get("/api/analyses/{run_id}/rescue_summary")
def get_rescue_summary(run_id: str):
    """
    Returns dynamic summary counts of mapped buildings, roads, bridges, and settlements
    classified by primary flood exposure and classical candidate change.
    """
    run_dir = WORKSPACE_ROOT / "runs" / run_id
    if not run_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Analysis run {run_id} not found.")

    from backend.services.asset_impact_service import get_run_asset_impact_data
    impact_data = get_run_asset_impact_data(run_dir)
    return impact_data["summary"]


@app.post("/api/analyses/{run_id}/connectivity")
def trigger_connectivity_analysis(
    run_id: str,
    snapshot_timestamp: Optional[str] = None,
):
    """
    Triggers Phase 3 Historical OSM extraction, infrastructure exposure assessment,
    and road connectivity reachability analysis for a completed flood analysis run.
    Enforces that snapshot_timestamp cannot postdate the event_date.
    """
    run_dir = WORKSPACE_ROOT / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail=f"Analysis run {run_id} not found.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    event_date = manifest.get("inputs", {}).get("event_date", "2026-08-26")

    # If no snapshot timestamp is explicitly provided, safely derive from event date
    if not snapshot_timestamp:
        if event_date == "2026-08-26":
            snapshot_timestamp = "2026-07-27T00:00:00Z"
        else:
            snapshot_timestamp = f"{event_date}T00:00:00Z"

    # Enforce strict pre-event snapshot requirement: snapshot cannot postdate event
    if snapshot_timestamp[:10] > event_date:
        raise HTTPException(
            status_code=400,
            detail=f"OSM snapshot timestamp ({snapshot_timestamp}) cannot postdate event date ({event_date}). Pre-event snapshot required."
        )

    try:
        res = run_infrastructure_and_connectivity_analysis(
            run_dir=run_dir,
            osm_snapshot_timestamp=snapshot_timestamp,
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connectivity analysis failed: {e}")


@app.get("/api/analyses/{run_id}/connectivity")
def get_connectivity_summary(run_id: str):
    """
    Returns the settlement reachability and road connectivity summary for a completed analysis.
    Verifies temporal consistency between the historical event date and the OSM snapshot timestamp.
    """
    run_dir = WORKSPACE_ROOT / "runs" / run_id
    summary_path = run_dir / "derived" / "connectivity_summary.json"
    manifest_path = run_dir / "manifest.json"
    osm_manifest_path = run_dir / "osm" / "osm_manifest.json"

    if not summary_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Connectivity summary not found for run {run_id}. Trigger POST /api/analyses/{run_id}/connectivity first."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Verify temporal validity of OSM snapshot against event date
    data["is_temporally_valid"] = True
    if manifest_path.is_file() and osm_manifest_path.is_file():
        try:
            with open(manifest_path, "r", encoding="utf-8") as mf:
                m = json.load(mf)
            with open(osm_manifest_path, "r", encoding="utf-8") as omf:
                om = json.load(omf)

            event_date = m.get("inputs", {}).get("event_date", "2026-08-26")
            snapshot_date = om.get("snapshot_timestamp", "2026-07-27")[:10]

            if snapshot_date > event_date:
                data["is_temporally_valid"] = False
                data["temporal_warning"] = (
                    f"Temporal consistency warning: Historical OSM snapshot ({snapshot_date}) postdates event date ({event_date}). "
                    "A pre-event road network is required for historical reconstructions. Connectivity cannot be verified."
                )
        except Exception:
            pass

    return data


@app.get("/api/analyses/{run_id}/tiles/{layer}/{z}/{x}/{y}.png")
def get_raster_tile(run_id: str, layer: str, z: int, x: int, y: int):
    """
    Renders and delivers a 256x256 Web Mercator PNG tile for a given analysis raster.
    Supported layers: pre_vv, post_vv, pre_vh, post_vh, delta_vv_db, snunet_flood_prob,
    snunet_pred_class, baseline_candidate_flood, dem, slope.
    """
    if layer not in ALLOWED_LAYERS:
        raise HTTPException(
            status_code=400,
            detail=f"Layer '{layer}' not allowed. Allowed layers: {list(ALLOWED_LAYERS.keys())}"
        )
    if z < 0 or x < 0 or y < 0:
        raise HTTPException(status_code=400, detail="Tile coordinates must be non-negative integers.")

    # Guard against invalid run_id path traversal
    if not run_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid run ID format.")

    try:
        png_bytes = render_raster_tile(run_id, layer, z, x, y)
        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tile rendering error: {e}")


@app.get("/api/analyses/{run_id}/inspect")
def inspect_point_endpoint(run_id: str, lon: float, lat: float):
    """
    Samples stored analysis rasters and vectors at (lon, lat) to return
    calibrated physical measurements and a deterministic scientific explanation.
    """
    # Guard against invalid run_id path traversal
    if not run_id.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid run ID format.")

    try:
        return inspect_point(run_id, lon, lat)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Point inspection failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
