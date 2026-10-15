# Flood Intelligence Platform — Track B: Mapping Flood Damage from Space

**Multimodal AI Hackathon 2026 — Track B**  
**Case Study:** Bhote Koshi–Trishuli River Corridor, Rasuwa / Nuwakot Districts, Nepal (August 2026 Flood Event)  
**Primary Observation Orbit:** Relative Orbit 19, Descending  
**Analysis Run ID:** `flood_20261014_082338_45889d`

---

## 1. Executive Overview

The **Flood Intelligence Platform** is a disaster-response remote-sensing and geospatial intelligence system designed to deliver rapid, automated, and scientifically verified flood inundation mapping and critical infrastructure exposure assessments from space.

Built around a deep Siamese neural network (**SNUNet-ECAM**) and multi-temporal Sentinel-1 Synthetic Aperture Radar (SAR) imagery, the platform:
1. **Acquires and Radiometrically Calibrates Spaceborne Radar:** Streams Radiometric Terrain-Corrected (RTC) Sentinel-1 observations before and after severe weather events.
2. **Predicts Inundation with Siamese Deep Learning:** Deploys SNUNet-ECAM to separate genuine flood inundation from permanent water and steep-terrain radar shadowing.
3. **Applies Independent Numerical Physics:** Computes decibel backscatter difference ($\Delta VV_{dB}$) and Copernicus DEM slope filtering as an unlearned classical baseline.
4. **Assesses Historical Infrastructure Exposure:** Intersects flood masks with pre-event OpenStreetMap infrastructure (highways, bridges, settlements, healthcare facilities) extracted from historical attic archives strictly prior to the disaster cutoff.
5. **Simulates Topological Settlement Reachability:** Constructs a metric routable graph across road networks to model post-flood emergency access and isolated hamlets using Dijkstra shortest-path evaluation.
6. **Delivers Real-Time Interactive Intelligence:** Streams dynamically reprojected Web Mercator raster tiles and coordinate-level physical radar measurements to an interactive map-first React dashboard supporting Swipe, Synchronized Side-by-Side, and Multi-Layer comparison modes.

---

## 2. Scientific Integrity & Disclosures

### A. Pretrained Model & Domain Gap Disclosure
- **Architecture:** SNUNet-ECAM (Siamese U-Net++ with Ensemble Channel Attention Module).
- **Pretrained Checkpoint:** Checkpoint loaded from Kuro Siwo reference weights (`checkpoints/best_segmentation.pt`), trained on multi-temporal SAR backscatter and Copernicus DEM (Bountos et al., NeurIPS 2024).
- **File Size & Checkpoint Hash:**
  - File Size: **144,657,509 bytes (~138 MB)**. Contains full PyTorch checkpoint dictionary (`epoch`, `model_state_dict` with 12,035,107 parameters, and AdamW optimizer states).
  - Verified SHA-256: `6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed`
- **Radiometric Domain Alignment:** Kuro Siwo training data was preprocessed from Sentinel-1 Ground Range Detected (GRD) products via ESA SNAP. Real-time inference executes on Sentinel-1 Radiometric Terrain Corrected (RTC) gamma-naught power mosaics from Microsoft Planetary Computer STAC. To bridge this domain gap without altering checkpoint weights:
  - Inputs are strictly verified as positive linear backscatter power.
  - Channels are normalized using the official Kuro Siwo global statistics ($VV_{mean} = 0.0953, VV_{std} = 0.0427$, $VH_{mean} = 0.0264, VH_{std} = 0.0215$, $DEM_{mean} = 67.03\text{ m}, DEM_{std} = 1765.01\text{ m}$).
  - Seamless 2D Hann window blending is applied across overlapping $224 \times 224$ inference patches (stride 168, 56px overlap) to eliminate border discontinuities.

### B. Analytical Independence & Reference Data Separation
- **No Reference Damage Maps in Inference:** Copernicus EMS Rapid Mapping activation **EMSR927** and published post-disaster damage polygons were **strictly excluded** from model inference, training, preprocessing, threshold tuning, and connectivity graphs.
- **Historical OSM Temporal Cutoff:** All infrastructure data (roads, bridges, settlements, hospitals) was queried from the OpenStreetMap Historical Attic Database with a strict cutoff timestamp of `2026-07-27T00:00:00Z`, ensuring post-event crowd-sourced edits did not leak into the pre-disaster baseline.
- **Modelled Reachability vs. Ground Truth:** Road segments intersecting flood predictions are classified strictly as `potentially_exposed` (not confirmed destroyed). Disconnected settlements represent topological network reachability models under simulated flooding, not field-verified ground closures.

### C. Road Exposure & Corridor-Level Closure Scenario
- **Submerged Road Length:** **190.1 meters** of aggregate road length intersects the modeled flood water across **12 unique OSM highway ways**.
- **Conservative Corridor-Level Closure Scenario (Baseline):**
  - In mountainous terrain, civil protection authorities typically issue whole-corridor closures when an unclassified highway is breached.
  - Under this operational scenario, all **1,436 graph edge fragments (26.14 km total length)** belonging to the 12 breached ways are removed from the routing graph (including 25.95 km outside the mapped flood polygon).
  - Reachability results: **2 settlements potentially cut off** (`Laharepauwa` and `Dhunge`), **2 detoured** (`Tupche` [+0.68 km] and `Uttargaya` [+4.42 km]), and **23 unaffected** (including `Betrawati` at 2.40 km baseline and post-event).
  - *Never describe these two settlements as confirmed real-world isolation; they reflect a conservative whole-way closure model.*
- **Localized Fragment-Level Traversal (Sensitivity Alternative):**
  - If only the **25 graph edge fragments (383.9 m)** that physically intersect the flood extent are severed, local bypasses remain intact.
  - Reachability results: **0 cut off**, **2 detoured** (`Tupche` [+0.68 km] and `Uttargaya` [+3.49 km]), and **25 unaffected** (with `Laharepauwa` and `Dhunge` remaining connected).

### D. Reconciled Scientific Area Statistics
- **2.87 ha (0.0287 km²):** Filtered, vectorized flood-water extent across the full unpadded AOI ($109.51\text{ km}²$), comprising 41 discrete polygons after applying the $500\text{ m}²$ minimum-area noise filter.
- **15.76 ha (0.1576 km²):** Raw, unfiltered Class 2 raster prediction across the full main-AOI processing grid ($1041 \times 1134$ at 10m = 1,576 pixels).
- **1.79 ha:** Raw Class 2 raster prediction within the $5.08\text{ km}²$ EMSR927 common evaluation footprint (179 pixels).
- **0.62 ha:** Filtered vectorized flood prediction within the EMSR927 common evaluation footprint (62 pixels).
- **10.7 ha:** Modeled permanent-water class (Class 1, rivers and reservoirs), strictly isolated as a baseline and excluded from flood metrics.
- **138.1 ha:** Classical radar-change candidate extent ($\Delta VV < -3\text{ dB}$, slope $\le 15^\circ$), capturing physical backscatter drop from surface smoothing and debris scouring.

### E. Copernicus EMSR927 Reference Benchmark Nuance
- Copernicus EMS activation **EMSR927** (Bidur, AOI03) is an emergency grading product mapping **"6-Mass Movement"** (landslide scouring and mudflows) from sub-meter optical sensors (BlackSky/Satellogic, 27 August 2026).
- Sentinel-1 SAR was acquired on 5 September 2026 (9 days later), after standing water largely receded.
- Consequently, this comparison is strictly an **independent cross-hazard spatial-overlap analysis**, NOT a direct flood-water segmentation ground truth validation. Classical radar drop ($\Delta VV < -3\text{ dB}$) captures 37.58 ha of the mass-movement corridor with 67.7% precision and 21.4% IoU. Do not present these metrics as validated flood-water accuracy.

### F. Optical Sentinel-2 Cloud Coverage Limitations
A systematic STAC query across Microsoft Planetary Computer for Sentinel-2 L2A optical acquisitions over the Trishuli AOI during the August–September 2026 monsoon window revealed pervasive cloud cover across all 15 available scenes (50.9% to 99.8% cloud cover; e.g. S2B on 27 August with 78.5% cloud cover). This empirical finding confirms why C-band Synthetic Aperture Radar (Sentinel-1 SAR) was engineered as the operational primary sensor: microwave radar penetrates dense monsoonal clouds where optical imagery suffers total occlusion.

---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        React / Leaflet Frontend                        │
│  - Swipe Comparison Divider  - Synchronized Side-by-Side Dual Maps     │
│  - Multi-Layer Overlay       - Clickable Point Inspector Drawer        │
│  - Reconciled Metric Cards   - Analysis Parameter Launch Modal         │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ HTTP REST / PNG Tiles / GeoJSON
┌─────────────────────────────────▼──────────────────────────────────────┐
│                        FastAPI Application (backend)                   │
├────────────────────────────────┬───────────────────────────────────────┤
│  Tile Streaming Engine         │  Point Inspector & Reasoning Engine   │
│  GET /tiles/{layer}/{z}/{x}/{y}│  GET /inspect?lon={lon}&lat={lat}     │
│  - Reprojects to EPSG:3857     │  - Samples 7 analytical rasters       │
│  - Bilinear continuous layers  │  - Queries nearest OSM infrastructure │
│  - Nearest-neighbour masks     │  - Generates deterministic reasoning  │
│  - Calibrated shared dB stretch│  - No LLM latency or hallucination    │
├────────────────────────────────┴───────────────────────────────────────┤
│  Pipelines: STAC Acquisition, SNUNet-ECAM Inference, Historical OSM    │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ Rasterio / PyTorch / NetworkX
┌─────────────────────────────────▼──────────────────────────────────────┐
│  Geospatial Storage & Analysis Runs (runs/)                           │
│  - Sentinel-1 RTC Pre/Post GeoTIFFs (2026-08-24 & 2026-09-05)          │
│  - Copernicus DEM & Slope GeoTIFFs (10m aligned)                       │
│  - SNUNet Probability, Class, & Baseline Change Rasters                │
│  - Historical OSM & Reachability Vector GeoJSONs                       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Mandatory Attributions & Data Citations

- **Copernicus Sentinel Data:**
  > "Contains modified Copernicus Sentinel data 2026."
  Data acquired via Microsoft Planetary Computer STAC API.
- **Copernicus Digital Elevation Model:**
  > Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- **OpenStreetMap Data:**
  > © OpenStreetMap contributors. Data available under the Open Database License (ODbL). Historical attic data accessed via HeiGIT ohsome API and Overpass API.
- **Neural Network Training Data & Architecture:**
  > Kuro Siwo: A global multi-temporal SAR dataset for rapid flood mapping (Bountos et al., NeurIPS 2024 Datasets and Benchmarks Track).

---

## 5. Built With & AI Tool Disclosures

### Core Technologies
- **Deep Learning:** PyTorch 2.6.0+cu124, torchvision, CUDA 12.4
- **Geospatial & Remote Sensing:** Rasterio, GeoPandas, Shapely, PyProj, STAC / pystac-client
- **Graph & Network Analysis:** NetworkX 3.6.1
- **Backend API:** FastAPI, Uvicorn, Pydantic, Pillow
- **Frontend Dashboard:** React 19, Vite, Leaflet, Lucide React

### AI Tools Disclosure
In compliance with Hackathon rules, AI coding assistance (Antigravity IDE / Google DeepMind pairing agent) was utilized during development for:
- Writing automated integration and regression test suites.
- Implementing Web Mercator bounding-box tile coordinate mathematics.
- Structuring React component layout and CSS glassmorphism styles.
- Drafting deterministic scientific explanation rules over physical radar measurements.

---

## 6. Verification & Automated Testing

The complete test suite contains **39 automated tests**:
- **37 Offline Unit & Integration Tests:** Cover GPU execution, model contracts, geospatial grid reprojection, slope physics, decibel conversions, OSM historical attic querying, network graph construction, Dijkstra reachability, tile streaming security, nearest-neighbour categorical resampling, active-run identity derivation, settlement connectivity consistency, individual asset exposure layers, plain-English point inspection headlines, and dynamic A4 situation report generation.
- **2 Live STAC Integration Tests:** Query live Microsoft Planetary Computer STAC for genuine multi-temporal Sentinel-1 RTC scenes and Copernicus DEM for the August 2026 case study and 2024 historical pairing.

```bash
# Run the fast offline test suite (37 tests, ~30s)
.venv\Scripts\pytest.exe tests/ --ignore=tests/test_real_integration.py -v

# Run the complete test suite including live satellite STAC queries (39 tests)
.venv\Scripts\pytest.exe tests/ -v
```

---

## 7. How to Run Locally

### Prerequisites
- Windows 10/11 or Linux
- Python 3.11+ (with virtual environment in `.venv`)
- Node.js 18+ and npm

### 1. Model Checkpoint Setup
> **Note on Checkpoint Storage:** To comply with GitHub's 100 MB per-file limit, `checkpoints/best_segmentation.pt` (~138 MB) is excluded from version control via `.gitignore`.
- On a fresh clone, ensure the checkpoint directory exists:
  ```powershell
  mkdir checkpoints
  ```
- Place the official Kuro Siwo `best_segmentation.pt` checkpoint file into `checkpoints/best_segmentation.pt`.
- Verify the SHA-256 integrity hash:
  ```powershell
  Get-FileHash checkpoints\best_segmentation.pt -Algorithm SHA256
  # Expected: 6c6bcb78d956f9d139743eb5bc44cf8e14528b643f9cbf5f4478f4e321a609ed
  ```

### 2. Start FastAPI Backend
```powershell
.venv\Scripts\activate
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger API documentation: `http://127.0.0.1:8000/docs`

### 3. Start React/Vite Frontend
```powershell
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
Dashboard UI: `http://127.0.0.1:5173/`

### 4. Build for Production
```powershell
cd frontend
npm run build
```
Generates optimized static bundle in `frontend/dist/`.
