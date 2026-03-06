# Citywide Crossing Width & Crosswalk Painting Quality (Philadelphia 2024 Aerial Imagery)  
**Scope:** Citywide deployment with local pilot first.  
**Primary outputs:** (1) Crossing width at intersection–approach level, (2) Crosswalk painting quality at crosswalk level.  
**Imagery:** PASDA Philadelphia Imagery 2024 (dataset 7031).  
**Reference geometry:** Street centerlines (OSM or City) used only for indexing/ROI, not as ground truth.

---

## 1) Checklist (Step-by-step)

### A. Project setup
- [ ] Define study boundary: Philadelphia city limits polygon (AOI).
- [ ] Choose reference street centerline source: **OSM** (fast) or **City centerline** (preferred if available).
- [ ] Generate **intersection points** and **approach segments** (street segments incident to each intersection).
- [ ] Decide output units: meters (recommended) and/or US feet (optionally both).

### B. Imagery access & preparation
- [ ] Download PASDA 2024 imagery tiles/mosaic covering the AOI.
- [ ] Build a tile index (or use provided index) and create a tiling scheme for processing.
- [ ] For each tile:
  - [ ] Ensure consistent CRS across imagery and reference vectors.
  - [ ] Validate pixel size and confirm linear units (feet vs meters).
  - [ ] Optional: apply radiometric normalization (e.g., histogram matching) to reduce tile-to-tile brightness shifts.

### C. ROI construction (critical for precision + efficiency)
- [ ] Create **Intersection ROIs**:
  - [ ] Buffer each intersection point (e.g., 25–60 m; set in pilot).
- [ ] Create **Approach ROIs** (for crossing width):
  - [ ] For each intersection–approach pair, construct a short “approach window” along the approach (e.g., 10–25 m from the intersection).
  - [ ] Within the window, generate multiple **perpendicular transects** (e.g., 10–30 lines) to sample crossing width.

### D. Local pilot (recommended first)
- [ ] Select 3–5 pilot subareas representing diverse street contexts:
  - [ ] Center City (high-rise shadows)
  - [ ] Rowhouse neighborhoods (dense grid)
  - [ ] Tree-canopy neighborhoods (occlusion)
  - [ ] Industrial/large paved surfaces (false positives risk)
- [ ] Create a **manual evaluation set**:
  - [ ] 50–100 intersection–approach pairs for width checks (visual/measure)
  - [ ] 200–500 crosswalks for quality rating (3- or 5-level human score)
- [ ] Finalize key parameters:
  - [ ] Intersection ROI radius
  - [ ] Transect count and spacing
  - [ ] Crosswalk detection thresholding and post-processing rules
  - [ ] Quality score component weights and QC flags

### E. Model & inference (citywide deployment)
**Goal 1 — Road surface mask (only for crossing width):**
- [ ] Train or apply a model that outputs a **road surface mask** (binary/probability).
- [ ] Constrain inference to **Intersection ROIs** (recommended) to avoid citywide road mapping.

**Goal 2 — Crosswalk detection/segmentation (for quality scoring):**
- [ ] Train or apply a model that outputs **crosswalk mask** (preferred) or bbox.
- [ ] Run inference only inside **Intersection ROIs**.

### F. Post-processing & metric extraction (the main deliverables)
**Crossing width (intersection–approach level):**
- [ ] For each intersection–approach:
  - [ ] Sample road mask along each perpendicular transect.
  - [ ] Compute the **continuous road-width segment length** on the transect.
  - [ ] Aggregate across transects:
    - [ ] `crossing_width = median(width_samples)`
    - [ ] `width_iqr = IQR(width_samples)`
    - [ ] `valid_samples = count(valid transects)`
  - [ ] Set QC flags (e.g., insufficient samples, extreme outliers, heavy occlusion).

**Crosswalk span & quality (crosswalk level):**
- [ ] Convert crosswalk mask to polygons (or refine bboxes to stripe regions).
- [ ] For each crosswalk:
  - [ ] Estimate **crosswalk span** (approx. crossing distance) using geometry principal axis or aligned measurement.
  - [ ] Compute quality components:
    - [ ] Contrast
    - [ ] Coverage
    - [ ] Continuity
    - [ ] Occlusion penalty
  - [ ] Combine into `quality_score_0_100` using calibrated weights.
  - [ ] Set `qc_flag_quality` if occlusion is too high or geometry is unreliable.

### G. Validation & QA (minimum recommended)
- [ ] Use **spatially separated** holdout areas for evaluation (avoid patch leakage).
- [ ] Width QA:
  - [ ] Compare a stratified sample of crossing widths against manual measurement.
- [ ] Quality QA:
  - [ ] Compare quality_score to human ratings; report rank correlation (e.g., Spearman).
- [ ] Error taxonomy:
  - [ ] Document common failure modes (tree canopy, shadows, parking lots, construction).

### H. Outputs & packaging
- [ ] Produce two primary geospatial layers:
  - [ ] `crossings` (intersection–approach): width metrics + QC
  - [ ] `crosswalks` (crosswalk): geometry + span + quality metrics + QC
- [ ] Export formats:
  - [ ] GeoPackage (`.gpkg`) or GeoParquet (recommended)
  - [ ] CSV summaries for tables/figures
- [ ] Save metadata:
  - [ ] imagery year, processing date, model version, parameter settings, CRS/unit notes

---

## 2) Methodology (Narrative)

### 2.1 Study objective
This project quantifies two pedestrian-relevant intersection attributes citywide using ultra–high-resolution aerial imagery:  
(1) **Crossing width** (the curb-to-curb crossing distance proxied by road-surface extent near intersections), and  
(2) **Crosswalk painting quality**, reflecting the visibility and integrity of marked crosswalk striping.

### 2.2 Data
The primary data source is Philadelphia 2024 orthorectified aerial imagery (PASDA dataset 7031). Street centerlines (from OSM or a municipal source) are used solely to create a consistent indexing system for intersections and approaches and to define analysis ROIs. The centerline data are not treated as ground-truth labels.

### 2.3 Intersection and approach indexing
Intersections are constructed from the street centerline network by identifying node locations where two or more segments meet. For each intersection, “approaches” are defined as the incident street segments. This yields an intersection–approach unit of analysis that supports street-level reporting and mapping.

### 2.4 ROI strategy
All inference and measurement are restricted to **Intersection ROIs** generated by buffering each intersection. This restriction substantially reduces false positives (e.g., parking lots, rooftops, large paved plazas) and improves computational efficiency. For crossing width, a secondary approach-specific window is defined along each approach to localize sampling and to align measurements with the street orientation.

### 2.5 Road-surface mask (for crossing width only)
A road-surface segmentation model is applied within Intersection ROIs to estimate the spatial extent of roadway pavement. The resulting binary/probability mask is not used to compute citywide road widths; it is used exclusively to estimate pedestrian-relevant crossing widths at intersections.

### 2.6 Crossing width estimation via transect sampling
For each intersection–approach, multiple perpendicular transects are cast across the roadway within the approach window near the intersection. Along each transect, the method measures the continuous span of road-surface mask pixels. The crossing width for that approach is then defined as the **median** span across transects, while the **interquartile range** captures uncertainty due to occlusions, local misclassification, and geometric variability. Quality-control flags are assigned when sample counts are low or when measurements are unstable.

### 2.7 Crosswalk detection/segmentation (for painting quality)
Crosswalks are detected or segmented within Intersection ROIs. Segmentation masks are preferred because they support pixel-based assessments of striping continuity and coverage. Detected crosswalk geometries are linked back to intersections and approaches via spatial association.

### 2.8 Crosswalk painting quality score
Crosswalk painting quality is quantified using a composite, interpretable scoring framework:
- **Contrast:** intensity difference between crosswalk striping and adjacent roadway, normalized to reduce sensitivity to tile brightness variation.
- **Coverage:** proportion of high-reflectance striping pixels within the crosswalk footprint, capturing fading.
- **Continuity:** fragmentation and gap statistics from the striping mask, capturing wear and partial repainting.
- **Occlusion penalty:** proportion of the crosswalk footprint affected by shadows or obstructions, used to down-weight unreliable scores.

These components are combined into a 0–100 score using weights calibrated in a pilot study with human ratings. Crosswalks with severe occlusion or unreliable geometry receive QC flags.

### 2.9 Pilot-to-citywide deployment
A local pilot is conducted to (i) tune ROI sizes and sampling densities, (ii) calibrate the quality score, and (iii) validate the crossing-width estimates. After parameter stabilization, the workflow is deployed citywide using a tile-based batch process with checkpointed intermediate outputs.

### 2.10 Validation
Validation follows spatially separated train/test splits to avoid overestimating performance. Crossing widths are validated against manual measurements for a stratified sample of intersections. Painting quality is validated against human ratings and summarized using rank correlation and confusion matrices for coarse classes (e.g., clear/moderate/faded).

### 2.11 Deliverables
The final deliverables are two GIS-ready layers:
1) **Crossings layer:** intersection–approach crossing widths with uncertainty and QC flags.  
2) **Crosswalks layer:** crosswalk geometries with span and painting quality scores plus QC flags.

Both layers are suitable for citywide mapping, street-level reporting, and downstream statistical analysis.

---
