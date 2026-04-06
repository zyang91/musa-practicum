# Streetscape Analysis Using AI
**Authors:** Zicheng Xiang, Zhanchao Yang  
**Advisors:** Dr. Xiaojiang Li & Dr. Erick Guerra  
Spring 2026, MUSA Practicum

---

## The Problem: A Data Gap in Philadelphia's Pedestrian Infrastructure

The City of Philadelphia currently lacks up-to-date, fine-grained data on pedestrian infrastructure at intersections. Key attributes — such as crosswalk width, crosswalk pavement condition, and the presence of traffic controls (stop signs, traffic signals) — are either outdated, incomplete, or captured only at the time of major capital projects.

Engineering-grade field surveys (e.g., LiDAR vehicles, manual measurement crews) are expensive, time-consuming, and typically only deployed when a major infrastructure upgrade is required to comply with federal or state policy. The City has no dedicated team to conduct routine, citywide pedestrian condition assessments. As a result, the Office of Transportation, Infrastructure, and Sustainability (OTIS) and other agencies must make investment and prioritization decisions using stale or coarse data.

**What's missing:**
- Curb-to-curb crossing distances at intersections across the city
- Crosswalk striping condition (faded, worn, or freshly painted)
- Locations of stop signs and streetlights at intersections

---

## Our Product: AI-Powered Intersection Attribute Extraction

We use **2024 high-resolution aerial imagery** (PASDA Philadelphia Imagery, ~15 cm/pixel) and **computer vision models** to automatically extract pedestrian-relevant attributes for every signalized intersection in Philadelphia — without field crews and at a fraction of the cost of traditional surveys.

Our product sits between existing GIS datasets and full engineering surveys:

| Data Type | Cost | Frequency | Detail Level |
|---|---|---|---|
| Existing city GIS data | Low | Irregular / outdated | Coarse |
| **Our product** | **Medium** | **Annual (imagery-driven)** | **Intersection-level** |
| Engineering LiDAR survey | Very high | Only at major upgrades | Millimeter-level |

We do **not** seek to replace engineering surveys. Instead, we provide a scalable, repeatable data layer to support microscale planning decisions — such as crosswalk repaving prioritization, signal upgrade targeting, and pedestrian safety audits.

### Outputs

1. **Crossing Width** — Curb-to-curb road width at each intersection approach, measured via perpendicular transect sampling on road-surface segmentation masks.
2. **Stop Sign & Streetlight Locations** — Point layers for traffic control features detected at intersections.

---

## Methodology

### Data
- **Imagery:** PASDA Philadelphia 2024 orthorectified aerial imagery (dataset 7031)
- **Reference geometry:** OSM street centerlines (used for intersection indexing and ROI construction only — not as ground truth)

### Processing Pipeline
1. **Intersection & Approach Indexing** — Derive intersection nodes and approach segments from the street network.
2. **Region-of-Interest (ROI) Construction** — Buffer each intersection to restrict all inference to relevant pavement areas, reducing false positives from parking lots, rooftops, and large plazas.
3. **Crossing Surface Segmentation** — Apply a segmentation model within each ROI to identify crossing lines.
4. **Crossing Width Measurement** — Cast perpendicular transects across each approach and compute the median road-surface span as the crossing width.
5. **Traffic Feature Detection** — Apply a YOLO-based object detector to identify stop signs and streetlights at each intersection.

---

## Final Product & Visualization

The project delivers two primary GIS layers:

- **Crossings layer:** intersection–approach crossing widths with uncertainty estimates and quality-control flags
- **Crosswalks layer:** crosswalk geometries with span and painting quality scores

These layers are packaged as GeoPackage (`.gpkg`) or GeoParquet files for use in city planning workflows.

An **interactive web application** will display results on a satellite basemap, allowing users to:
- Click any intersection to view crossing width, crosswalk quality score, and nearby traffic controls
- Launch Google Street View directly from the map for visual verification
- Explore stop sign and streetlight locations citywide
