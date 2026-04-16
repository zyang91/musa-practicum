# Streetscape Analysis Using AI: Mapping Crosswalk Conditions and Pedestrian Infrastructure from Streetview and Arieal Imagery
**Authors:** Zicheng Xiang, Zhanchao Yang  
**Advisors:** Dr. Xiaojiang Li & Dr. Erick Guerra  
Spring 2026, MUSA Practicum


## Overview

This project develops a scalable, image-based pipeline for extracting pedestrian infrastructure attributes at intersections in Philadelphia. Using high-resolution imagery and deep learning models, we identify crosswalk locations, estimate crossing distances, and detect traffic control features, including stop signs and streetlights, across 4 neighborhood and district across city of Philadelphia. The work is exploratory and descriptive. Our primary contribution is a proof workflow that demonstrates how computer vision can help fill a persistent data gap in urban planning practice, particularly for pedestrian safety and walkability assessment with a low cost alternative.


## Motivation: Crosswalks as a Dimension of Urban Walkability

Walkability is a multidimensional concept that encompasses street connectivity and land use mix. Walking score is one of the key matrics for people to measure their neighborhood walkibility. According the walking score calculation matrics, walk scores only analyzes hundreds of walk routes to nearby amenities. Points are awarded based on the distance to amenities in each category. Walking scores also measures pedestrain friendlness by analyzing population density and road metrics such as block length and intersection density. However, the safety and quality of the pedestrian environment at street crossings are always out of consideration. Without adequate and proper safe crossing intersection, pedestain cannnot choose the closest route when travel from point A to point B.

Crosswalks are a key point of intersection between pedestrian and vehicular movement. Their physical condition and visibility directly affect pedestrian safety. Wide crossing distances increase exposure time for pedestrians, especially older adults and children. Faded or poorly maintained crosswalk markings reduce visibility for both drivers and pedestrians, increasing the risk of conflict. Despite these known relationships, systematic, citywide data on crosswalk conditions — particularly crossing width and marking quality — remain largely unavailable in many U.S. cities, including Philadelphia.

Existing pedestrian infrastructure datasets in the City of Philadelphia are either aggregated at a coarse level, collected irregularly through capital project workflows, or not maintained with sufficient frequency to reflect current conditions on the ground. The OTIS and other planning agencies currently lack a repeatable, low-cost method for tracking fine-grained intersection-level pedestrian attributes at scale.

**What's missing in current data:**
- Curb-to-curb crossing distances at intersections across the city
- Crosswalk striping condition (faded, worn, or freshly painted)
- Locations of stop signs and streetlights at intersections

This project is designed to address that gap by proposing and testing a computer vision based approach that can be applied at the scale of a full city using readily available aerial imagery and street view imagery.


## Contribution: Bridging the Gap Between GIS Data and Engineering Surveys

City planning currently relies on two types of pedestrian infrastructure data: coarse GIS datasets that are updated infrequently, and expensive engineering surveys deployed only at the time of major capital upgrades. There is no intermediate, routinely updated data source that provides intersection-level pedestrian attributes at citywide scale.

This project proposes a workflow that occupies that middle position — offering more spatial detail and thematic specificity than existing GIS layers while remaining far more scalable and lower in cost than field-based engineering measurement. Our approach uses aerial imagery that is already captured annually by the Pennsylvania Spatial Data Access (PASDA) program, meaning the pipeline can in principle be reapplied without new data collection as imagery is refreshed over time.

| Data Type | Cost | Frequency | Detail Level |
|---|---|---|---|
| Existing city GIS data | Low | Irregular / outdated | Coarse |
| **This workflow** | **Medium** | **Annual (imagery-driven)** | **Intersection-level** |
| Engineering LiDAR survey | Very high | Only at major upgrades | Millimeter-level |

We do **not** claim that this workflow replaces engineering surveys or produces measurement-grade outputs. Rather, it is intended as a screening and prioritization tool — one that can help direct attention and resources toward intersections that may warrant closer inspection or targeted investment.

### Outputs

1. **Crossing Distance** — Curb-to-curb road width at each intersection approach, estimated via perpendicular transect sampling on road-surface segmentation masks.
2. **Crosswalk Detection** — Predicted crosswalk polygon locations derived from U-Net segmentation of aerial imagery, providing spatial coverage and marking visibility.
3. **Stop Sign & Streetlight Locations** — Point layers for traffic control features detected at intersections using a fine-tuned YOLO object detection model.

---

## Methodology

### Data
- **Imagery:** PASDA Philadelphia 2024 orthorectified aerial imagery (dataset 7031, ~15 cm/pixel)/ Google Arieal imagery
- **Reference geometry:** OpenStreetMap (OSM) street centerlines, used for intersection indexing and region-of-interest construction only — not as ground truth for model evaluation

### Models
- **U-Net segmentation** — A semantic segmentation model trained on manually labeled crosswalk features to identify crosswalk pixels in aerial imagery. The model was trained on image-mask patch pairs derived from a labeled dataset of 202 crosswalk features in the University City study area.
- **YOLO object detection** — A fine-tuned YOLO model applied to detect stop signs and streetlights at intersections. Labels were prepared to support domain-specific fine-tuning on Philadelphia street imagery.

### Processing Pipeline
1. **Intersection & Approach Indexing** — Derive intersection nodes and approach segments from the street network.
2. **Crosswalk Surface Segmentation** — Apply the U-Net model within each ROI to generate binary predictions of crosswalk pixels.
3. **Crossing Distance Measurement** — Cast perpendicular transects across each approach and compute the median road-surface span as the estimated crossing width.
4. **Traffic Feature Detection** — Apply the YOLO detector to identify stop signs and streetlights at each intersection.

This is an early-stage, descriptive workflow. Results should be interpreted as a first step toward developing a more complete pedestrian infrastructure monitoring system, not as a final or validated planning product.


## Final Product & Visualization

The project delivers two primary GIS layers:

- **Crossings layer:** intersection–approach crossing distances with quality-control flags
- **Crosswalks layer:** predicted crosswalk polygon geometries with marking coverage estimates

These layers are provided as GeoPackage (`.gpkg`) or GeoParquet files suitable for use in city planning workflows.

An **interactive web application** displays results on a satellite basemap, allowing users to:
- Click any intersection to view estimated crossing width, crosswalk presence, and nearby traffic controls
- Launch Google Street View directly from the map for visual verification
- Explore stop sign and streetlight locations citywide
