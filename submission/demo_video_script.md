# 3-Minute Video Demonstration Script & Production Guide

**Project:** Multimodal AI Hackathon 2026 — Track B: Mapping Flood Damage from Space  
**Video Title:** Rapid Geospatial Flood Damage & Connectivity Intelligence from Space  
**Target Duration:** Exactly 3:00 (180 seconds)  
**Primary Case Study:** Trishuli River Corridor, Nepal (August 2026 Flood Event)  
**Analysis Run ID:** `flood_20261014_082338_45889d`

---

## Storyboard & Timed Narration Breakdown

### Phase 1: Problem Definition & Operational Context (0:00 – 0:25 | 25s)
- **Visual:** Browser viewport showing the Flood Intelligence Platform dashboard at `http://localhost:5173`. Header displaying Orbit 19 Descending and Verified Case Study (2026-08-26).
- **Narration:**
  > "During extreme monsoon events like the August 26, 2026 flash floods in Nepal's Trishuli corridor, cloud cover blinds optical satellites. Emergency commanders cannot wait days for clear skies to answer three critical questions:
  > 1. Where is the active flood water?
  > 2. Which highway corridors are submerged?
  > 3. Which isolated communities are cut off from medical care?
  > Our platform automates cloud-penetrating Sentinel-1 SAR acquisition, deep Siamese segmentation, and network reachability to deliver rapid, scientifically verified answers within minutes."

---

### Phase 2: Genuine SAR Observations & Swipe Comparison (0:25 – 1:05 | 40s)
- **Visual:** User drags the interactive Swipe Comparison divider back and forth across the Trishuli valley. Floating badges display `PRE-EVENT: 2026-08-24` and `POST-EVENT: 2026-09-05`. User switches briefly to Side-by-Side mode showing synchronized pan/zoom.
- **Narration:**
  > "Rather than using synthetic imagery or differing orbits, we strictly pair observations from Sentinel-1 Relative Orbit 19 descending.
  > On the left: August 24, 2026 pre-event SAR backscatter.
  > On the right: September 5, 2026 post-event pass.
  > Both rasters share an identical calibrated dB display stretch (-25 to -3 dB) so apparent differences reflect genuine physical surface changes.
  > Notice the river gorge: dark radar returns indicate surface smoothing and water accumulation, while mountain ridges reflect radar backscatter away from the sensor."

---

### Phase 3: Neural Segmentation, Physics Baseline & Point Inspection (1:05 – 1:40 | 35s)
- **Visual:** Switch to Multi-Layer Overlay mode. Toggle on SNUNet Flood Water (electric blue: 2.87 ha), Permanent River (cyan), and Classical Candidate Change (amber: 138.1 ha). Click on an inundated river terrace point (lon: 85.20, lat: 28.00). Inspector drawer opens on the right displaying physical measurements and deterministic reasoning.
- **Narration:**
  > "We separate deep neural predictions from classical physics.
  > In electric blue: our SNUNet Siamese neural network segments 2.87 hectares of active standing water along the lower terraces, discarding mountain shadow.
  > In amber: our classical baseline identifies 138.1 hectares with radar backscatter drops exceeding -3 dB.
  > When an operator clicks any point—such as this terrace at longitude 85.20, latitude 28.00—our backend samples 7 stored rasters instantly.
  > The system outputs a deterministic scientific explanation: a 2.86 dB backscatter drop is measured, but because terrain slope is 27.5 degrees, it exceeds the slope filter and SNUNet classifies it as dry land with 0% flood probability. Zero LLM hallucinations; pure physical reproducibility."

---

### Phase 4: Infrastructure Exposure & Isolated Settlement Modeling (1:40 – 2:15 | 35s)
- **Visual:** Zoom into the highway network and settlement markers. Red lines show the 12 potentially exposed road ways. Red settlement markers highlight Laharepauwa and Dhunge.
- **Narration:**
  > "For rescue routing, we extracted historical OpenStreetMap infrastructure prior to the disaster cutoff date of July 27, 2026.
  > By intersecting our flood mask with road vectors, we identify 12 unique OSM road ways with 190.1 meters of direct inundation.
  > Under our Conservative Corridor-Level Closure Scenario—where entire highway ways are assumed closed when any section intersects flood water—1,436 graph edges totalling 26.14 kilometers are removed to model corridor disruption.
  > Under this conservative scenario, two settlements—Laharepauwa and Dhunge—are potentially cut off from their primary hospital route, while Tupche and Uttargaya face detours. Under strict fragment-level removal, zero settlements are cut off. We never claim confirmed real-world bridge destruction or physical isolation."

---

### Phase 5: Independent EMSR927 Reference Benchmark (2:15 – 2:40 | 25s)
- **Visual:** Display the publication-quality comparison graphic (`submission/emsr927_validation_comparison.png`) showing EMSR927 Mass Movement vs. Classical Baseline vs. SNUNet.
- **Narration:**
  > "To validate our findings, we benchmarked our predictions against official Copernicus Emergency Management Service activation EMSR927.
  > Because this Himalayan event was a glacial lake outburst flood, Copernicus photo-interpreters mapped 157.6 hectares of high-gradient debris flows and landslide scars.
  > Our classical radar change layer captured 37.6 hectares of this debris corridor with 67.7% precision and 21.4% IoU.
  > Meanwhile, SNUNet targets standing flood water rather than dry mass movement, proving the critical value of combining physical radar change detection with neural water segmentation."

---

### Phase 6: Conclusion, Limitations & Reproducibility (2:40 – 3:00 | 20s)
- **Visual:** Display the One-Page Situation Report PDF, summary cards, and GitHub repository.
- **Narration:**
  > "In summary: a complete, cloud-penetrating flood intelligence pipeline running on genuine spaceborne radar, backed by 39 passing automated tests (37 offline, 2 STAC integration), a verified single-page situation report, and full Copernicus attributions.
  > All code, tests, and reproducibility scripts are available on our repository. Thank you."

---

## Production Artifact & Local Recording Reference
- **Compiled Master Demonstration Video:** [demo_video.mp4](file:///c:/Users/nikit/OneDrive/Desktop/Multi_modelHackathom/submission/demo_video.mp4)
- **Verified Duration:** 180.00 seconds (exactly 3:00 minutes)
- **Format:** H.264 / MP4 (1920x1088, 15.0 fps, 2,700 frames, 5.18 MB)
- **Pipeline Script:** [scripts/record_demo_video.py](file:///c:/Users/nikit/OneDrive/Desktop/Multi_modelHackathom/scripts/record_demo_video.py)
- **Live Local Recording (Alternative / Voiceover Capture):**
  1. Ensure the backend (`uvicorn backend.main:app --port 8000`) and frontend (`npm run dev --port 5173`) are running.
  2. Open Google Chrome or Microsoft Edge at `http://127.0.0.1:5173/` in fullscreen (F11 or 1920x1080 window).
  3. Use Windows Game Bar (`Win + Alt + R`) or OBS Studio to record voiceover against the storyboard sequence.
