# FloodLens — Final Submission Consistency & Verification Report

**Project:** Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space  
**System Name:** FloodLens: Rapid Satellite Flood Intelligence & Connectivity  
**Analysis Run ID:** `flood_20261014_082338_45889d` (Trishuli River Corridor, Nepal)  
**Date Generated:** 2026-10-15  
**Audit Status:** PASSED (All deliverable constraints, test suites, and empirical claims reconciled)

---

## 1. Files Changed & Reason for Each

| File | Changes Made | Scientific & Methodological Rationale |
| :--- | :--- | :--- |
| **`README.md`** | Updated Section 2, Section 6, and Section 7. | • Defined **Conservative Corridor-Level Closure Scenario** (190.1 m submerged across 12 OSM ways; 1,436 edges / 26.14 km removed; 2 cut-off, 2 detoured, 23 unaffected) and **Localized Fragment Sensitivity** alternative (25 edges / 383.9 m; 0 cut-off).<br>• Reconciled scientific areas (2.87 ha sieved vector, 15.76 ha raw raster, 1.79 ha raw in EMSR927, 0.62 ha vector in EMSR927, 10.7 ha permanent river, 138.1 ha classical change).<br>• Defined EMSR927 exclusively as optical mass-movement reference for cross-hazard spatial overlap analysis.<br>• Documented verified checkpoint size (~138 MB) and SHA-256 hash.<br>• Updated test suite documentation to 39 tests (37 offline, 2 STAC). |
| **`submission/one_page_situation_report.html`** | Updated metric badges, settlement accessibility table, infrastructure summary, and methodological disclosures. | • Explicitly labeled settlement reachability as **Conservative Corridor Closure**.<br>• Updated table with active run values (`Laharepauwa` 5.31 km, `Dhunge` 3.08 km, `Tupche` 4.99 km [+0.68 km], `Uttargaya` 7.37 km [+4.42 km], `Betrawati` 2.40 km unaffected).<br>• Contrasted 190.1 m submerged road with 26.14 km corridor closure and 383.9 m fragment sensitivity. |
| **`submission/one_page_situation_report.pdf`** | Recompiled from updated HTML via headless Chromium. | • Verified page count: **exactly 1 page** (2,510,572 bytes).<br>• Preserves all mandatory Copernicus, DEM, OSM, and Kuro Siwo attributions. |
| **`submission/technical_report.html`** | Updated Section 2.4, 3.2, 3.3, 3.4, 4.2, 4.4, Figure 2 caption, Section 6.2 callout, and adjusted Figure 1 max-height. | • Reconciled all 6 area figures across Section 2.4 and 3.4.<br>• Clarified EMSR927 cross-hazard spatial-overlap caveat in Section 3.4.<br>• Reconciled routing metrics to Conservative Corridor Closure Scenario vs. Fragment Sensitivity in Section 4.2 & 4.4.<br>• Documented checkpoint size (~138 MB) and SHA-256 hash in Section 3.2.<br>• Updated test suite outcome to 39 tests (37 offline, 2 STAC) in Section 6.2.<br>• Adjusted Figure 1 max-height to 52mm so Section 3.4 text renders with zero clipping. |
| **`submission/technical_report.pdf`** | Recompiled from updated HTML via headless Chromium. | • Verified page count: **exactly 6 pages** (3,253,510 bytes).<br>• No clipped text, unreadable tables, or missing caveats. |
| **`frontend/src/App.jsx`** | Updated Card 2 (Road Infrastructure Impact), Card 3 (Settlement Accessibility), Side Panel drawer notes, and Settlement Explainer section. | • Card 2 displays "Corridor Edges Removed: 1,436 (26.14 km)" alongside "190.1 m submerged (Not confirmed destroyed)".<br>• Card 3 displays status chip `CORRIDOR SCENARIO` and footnotes conservative corridor closure assumption.<br>• Side Panel and Explainer section document that 2 cut-off hamlets reflect whole-way corridor closure, noting that 0 settlements are cut off under fragment-level removal.<br>• Never implies real-world confirmed bridge destruction. |
| **`frontend/dist/`** | Rebuilt production bundle via `npm run build`. | • Built in 1.36s: `dist/index.html` (0.45 kB), `dist/assets/index-DN2pOWm9.css` (41.59 kB), `dist/assets/index-CaqyPBes.js` (463.48 kB). |
| **`submission/demo_video_script.md`** | Updated Phase 4 and Phase 6 narration. | • Phase 4 narration incorporates Conservative Corridor-Level Closure Scenario, 190.1 m submerged road across 12 ways, 1,436 edges / 26.14 km removed, and fragment sensitivity alternative (0 cut-offs).<br>• Phase 6 narration incorporates verified 39 tests (37 offline, 2 STAC). |

---

## 2. Confirmed Test Results

All tests executed in the repository's `.venv` environment (Python 3.13.0, PyTorch, Rasterio, Shapely, NetworkX, FastAPI):

| Test Suite Component | Execution Command | Result | Duration | Scope Covered |
| :--- | :--- | :--- | :--- | :--- |
| **Offline Test Suite** | `.venv\Scripts\pytest.exe tests/ --ignore=tests/test_real_integration.py -v` | **37 PASSED** | 54.81s | GPU execution, model contracts, coordinate transforms, 2nd-order slope calculation, decibel conversions, OSM historical attic queries, network graph construction, Dijkstra shortest-path reachability, Web Mercator tile math, path traversal protection, nearest-neighbour categorical resampling, active run consistency, asset impact layers, plain-English point inspector headlines, and dynamic A4 situation report generation. |
| **Live STAC Integration** | `.venv\Scripts\pytest.exe tests/test_real_integration.py -v` | **2 PASSED** | 124.49s | Live anonymous queries to Microsoft Planetary Computer STAC discovering Sentinel-1 RTC scenes (Relative Orbit 19 Descending) and Copernicus DEM 30m for the August 2026 challenge case study and 2024 historical regression pairing. |
| **Total Test Suite** | `.venv\Scripts\pytest.exe tests/ -v` | **39 PASSED (100%)** | ~179s | Complete codebase verification. Zero failures, zero skips. |

---

## 3. Deliverable Constraints & Verification

| Deliverable | Requirement Constraint | Verified Measurement | Compliance Status |
| :--- | :--- | :--- | :--- |
| **Situation Report** | Exactly 1 page | **1 page** (2,510,572 bytes) | **PASS** (Confirmed via regex page tree & visual inspection) |
| **Technical Report** | No more than 6 pages | **6 pages** (3,253,510 bytes) | **PASS** (Confirmed via regex page tree & visual inspection) |
| **Demo Video** | No more than 180 seconds | **180.00 seconds** (2,700 frames @ 15.0 fps) | **PASS** (Confirmed via OpenCV frame/FPS verification) |
| **Video Dimensions** | HD / High Quality | 1920 &times; 1088, H.264, 5.18 MB | **PASS** (Self-contained, web-optimized MP4) |

---

## 4. Checkpoint Provenance, Hash & Download Protocol

| Property | Verified Value |
| :--- | :--- |
| **Model Architecture** | SNUNet-ECAM (Siamese U-Net++ with Ensemble Channel Attention Module) |
| **Training Dataset** | Kuro Siwo: Global multi-temporal SAR dataset for rapid flood mapping (Bountos et al., NeurIPS 2024) |
| **File Location** | `checkpoints/best_segmentation.pt` |
| **File Size** | **144,657,509 bytes (~138 MB)** (12,035,107 parameters + AdamW optimizer states) |
| **Verified SHA-256** | `6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed` |
| **Git Exclusion** | Excluded in `.gitignore` (`checkpoints/`, `*.pt`) to comply with GitHub's 100 MB per-file limit |

### Clean-Clone Setup Instructions for Maintainer
1. On a fresh clone, create the checkpoint directory:
   ```powershell
   mkdir checkpoints
   ```
2. Place the official Kuro Siwo `best_segmentation.pt` weight file into `checkpoints/best_segmentation.pt`.
3. Verify integrity:
   ```powershell
   Get-FileHash checkpoints\best_segmentation.pt -Algorithm SHA256
   # Expected: 6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed
   ```
4. **Stable Download Method Note:** Because the Kuro Siwo NeurIPS 2024 benchmark authors distribute model weights via academic repositories (Hugging Face / Zenodo) requiring authentication or ephemeral signed tokens, we do **not** fabricate an artificial release URL or bake temporary tokens into the repository. The maintainer must supply the genuine checkpoint and verify the expected SHA-256 hash.

---

## 5. Final Git Readiness Check

| Category | Audit Finding | Status |
| :--- | :--- | :--- |
| **Secrets & Credentials** | No `.env` files, API keys, credentials, or SAS tokens are tracked. All STAC queries use anonymous public access with dynamic token signing. | **CLEAN** |
| **Virtual Environments** | `.venv/` and `venv/` are excluded via `.gitignore`. | **CLEAN** |
| **Model Weights** | `checkpoints/` and `*.pt` (138 MB) are excluded via `.gitignore`. No file exceeds GitHub's 100 MB limit. | **CLEAN** |
| **Raster Data & Runs** | `runs/`, `data/`, `scratch/`, and `*.tif` are excluded via `.gitignore`. | **CLEAN** |
| **Frontend Builds** | `frontend/dist/` and `frontend/node_modules/` are excluded via `.gitignore`. | **CLEAN** |
| **Deliverables Size** | `submission/demo_video.mp4` is 5.18 MB; PDFs are 2.51 MB and 3.25 MB. All well below GitHub limits. | **CLEAN** |
| **Files to Stage** | Root `README.md`, `.gitignore`, `backend/`, `frontend/` (source code and configs), `scripts/`, `submission/`, `tests/`. | **READY** |
| **Git Actions Withheld** | Per instructions, **no Git initialization, remote creation, commit, or push** was performed by the agent. | **COMPLIANT** |

---

## 6. Unresolved Claims That Must NOT Be Made in Devpost Description

To maintain strict scientific accuracy and compliance during judging, the team must **never** make the following claims:

1. **Never claim 26.14 km of roads are submerged:** Only **190.1 meters** of road geometry directly intersects the modeled flood water across 12 unique OSM ways. 26.14 km represents the cumulative length of all 1,436 graph edge fragments belonging to those 12 ways removed under the **Conservative Corridor-Level Closure Scenario**.
2. **Never claim Laharepauwa and Dhunge are confirmed real-world isolated:** They are modeled as cut off strictly under the whole-way closure assumption. Under localized fragment-level removal, **0 settlements are cut off**. Always frame this as an operational scenario rather than physical destruction.
3. **Never claim local model training or fine-tuning:** The system performs zero-shot inference using the official pretrained Kuro Siwo SNUNet-ECAM checkpoint with global normalization.
4. **Never present EMSR927 metrics as validated flood segmentation accuracy:** EMSR927 is an optical mass-movement grading product (landslide and debris flows mapped on 27 August), not standing water ground truth. Its overlap metrics evaluate physical co-occurrence with event deposits.
5. **Never claim zero building damage across the district:** Zero exposed buildings applies strictly to the 1,000-feature historical OSM sample queried within the AOI, not the entire municipality.
