## Method

This project developed a Python-based workflow to estimate crossing-related street geometry from high-resolution aerial imagery in the University City pilot area. First, road-surface labels were rasterized from manually prepared training data, and the imagery was split into image-mask patches for semantic segmentation. A U-Net model was then trained on these patches to identify road surface pixels from the aerial mosaic. The final model achieved a best validation Dice score of 0.9088. Threshold tuning on the test set identified 0.45 as the best binarization threshold, yielding a Dice/F1 score of 0.9530 and an IoU of 0.9103.

After model training, the final U-Net was applied to the full pilot mosaic using a sliding-window prediction approach. The resulting binary road-surface raster was cleaned through simple post-processing and polygonized into vector road-surface features. To estimate crossing distance, the workflow then combined the predicted road-surface polygons with OSM-derived intersection and edge data. For each intersection leg, a measurement line approximately perpendicular to the roadway was generated and intersected with the predicted road-surface polygon. The length of the resulting segment was used as an estimate of curb-to-curb crossing distance.

## Findings

The full-scene road-surface prediction was visually acceptable in ArcGIS and produced a conservative road mask with low false positive rates. Although some road areas were missed, the overall spatial pattern of the predicted surface was usable for exploratory geometric measurement. The final cleaned road-surface output was converted into 1,101 polygons.

Using the automated crossing-distance workflow, the model generated 93 valid crossing-distance estimates. The estimated crossing distances had a mean of 32.1 feet and a median of 25.9 feet. The interquartile range extended from 15.6 feet to 44.5 feet, and the maximum observed value was 89.7 feet. These values fall within a plausible range for urban street crossings and show that the workflow can generate intersection-level geometric measures from image-derived road-surface predictions.

## Limitations

Several limitations should be noted. First, the road-surface segmentation was intentionally conservative: false positives were low, but some true road areas were not captured. As a result, the estimated crossing distances are likely to be biased downward in some cases. In other words, these measurements should be interpreted as approximate lower-bound estimates rather than exact field measurements.

Second, the crossing-distance workflow does not directly identify marked crosswalks or curb ramps. Instead, it estimates crossing distance from predicted road-surface extent and OSM-based intersection geometry. This means the outputs are best understood as intersection-leg-level approximations of curb-to-curb width, not exact pedestrian crossing distances at specific marked crossings.

Third, the workflow depends on the quality of both the segmentation output and the OSM street network. Errors in either source can affect final measurements. Some intersection legs were also not successfully measured, so coverage remains incomplete. For these reasons, the results are most appropriate for pilot analysis, exploratory screening, and proof-of-concept work rather than engineering-grade design measurement.


## How crossing distance is measured

This workflow estimates crossing distance at the **intersection-leg level** using three inputs:

1. predicted road-surface polygons from the segmentation model  
2. OSM-derived intersection points  
3. OSM road edges connected to each intersection  

For each intersection, the script identifies nearby road legs from the OSM network. It then estimates the direction of each leg and creates a short measurement line that is **approximately perpendicular to the roadway**. This line is placed slightly away from the intersection center so that it crosses the road surface rather than the middle of the intersection node.

The script intersects that measurement line with the predicted road-surface polygon. The length of the intersected segment is treated as the **crossing distance**, which is interpreted as an approximate **curb-to-curb width** at that leg.

This means the output is **not** a manually measured marked crosswalk length. It is an automated geometric estimate based on segmented road surface and OSM intersection structure.

## How to use the outputs

The workflow produces three main outputs:

### `crossing_distance_points.gpkg`
Use this for **mapping** crossing distance locations.  
Each point represents one valid crossing-distance estimate, and the attribute `crossing_ft` stores the estimated distance in feet.

Best use:
- thematic maps
- graduated color maps
- spatial comparison of shorter vs. longer crossings

### `crossing_distance_lines.gpkg`
Use this for **method visualization**.  
Each line is the actual segment used to measure crossing distance.

Best use:
- showing how the estimate was generated
- overlaying on imagery or road polygons
- checking whether the measurement looks reasonable

### `crossing_distance_table.csv`
Use this for **summary statistics and charts**.  
This table summarizes crossing-distance estimates by intersection.

Best use:
- descriptive statistics
- histograms or boxplots
- comparing distributions across sites

## Important interpretation notes

- Distances are estimated **by leg**, not by assigning one single value to the whole intersection.
- A four-way intersection does **not automatically mean four identical crossing distances**.
- Because the road-surface segmentation is conservative, some distances may be slightly **underestimated**.
- These outputs are appropriate for **pilot analysis, exploratory measurement, and visualization**, not engineering-grade field measurement.
