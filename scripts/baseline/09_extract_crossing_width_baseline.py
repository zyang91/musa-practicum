import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pathlib import Path

# paths
mosaic_path = "../../data/processed/university_city_pilot_mosaic.tif"
transects_path = "../../data/vector/crossing_transects.gpkg"
output_dir = Path("../../data/output")
output_dir.mkdir(parents=True, exist_ok=True)

# parameters
sample_step_ft = 1.0          # sample every 1 foot along each transect
center_window_ft = 40.0       # only use central 80 ft region for baseline width
brightness_threshold = 170    # crude threshold for bright pavement / marking area

# read transects
transects = gpd.read_file(transects_path)

print("Loaded transects:", len(transects))

results = []

with rasterio.open(mosaic_path) as src:
    transform = src.transform
    width = src.width
    height = src.height

    for idx, row in transects.iterrows():
        line = row.geometry

        if line is None or line.is_empty or line.length <= 0:
            continue

        # sample points along transect
        n_steps = max(2, int(line.length // sample_step_ft))
        dists = np.linspace(0, line.length, n_steps)

        vals = []
        offsets = []

        for d in dists:
            pt = line.interpolate(d)
            x, y = pt.x, pt.y

            r, c = rowcol(transform, x, y)

            if r < 0 or r >= height or c < 0 or c >= width:
                vals.append(np.nan)
                offsets.append(d - line.length / 2.0)
                continue

            # read one pixel from each RGB band
            pix = src.read([1, 2, 3], window=((r, r + 1), (c, c + 1)))
            pix = pix[:, 0, 0].astype(float)

            # brightness proxy
            brightness = pix.mean()

            vals.append(brightness)
            offsets.append(d - line.length / 2.0)

        vals = np.array(vals)
        offsets = np.array(offsets)

        # restrict to center window around transect midpoint
        keep = np.abs(offsets) <= center_window_ft
        vals_center = vals[keep]
        offsets_center = offsets[keep]

        if len(vals_center) < 5:
            width_ft = np.nan
        else:
            # crude rule: pixels darker than threshold often correspond better to asphalt road
            # this is only a baseline heuristic
            road_like = vals_center < brightness_threshold

            # find longest continuous True run
            max_run = 0
            current_run = 0

            for flag in road_like:
                if flag:
                    current_run += 1
                    max_run = max(max_run, current_run)
                else:
                    current_run = 0

            width_ft = max_run * sample_step_ft

        results.append(
            {
                "transect_id": row["transect_id"],
                "approach_id": row["approach_id"],
                "intersection_id": row["intersection_id"],
                "transect_no": row["transect_no"],
                "offset_ft": row["offset_ft"],
                "width_ft_baseline": width_ft,
                "geometry": line,
            }
        )

transect_widths = gpd.GeoDataFrame(results, geometry="geometry", crs=transects.crs)

# save transect-level output
transect_out = output_dir / "transect_widths_baseline.gpkg"
transect_widths.to_file(transect_out, driver="GPKG")

print("\nSaved transect-level baseline:")
print(transect_out)

# aggregate to approach level
def iqr(arr):
    arr = np.array(arr)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    return np.percentile(arr, 75) - np.percentile(arr, 25)

agg = (
    transect_widths.groupby(["approach_id", "intersection_id"])
    .agg(
        crossing_width_ft=("width_ft_baseline", "median"),
        width_iqr_ft=("width_ft_baseline", iqr),
        valid_samples=("width_ft_baseline", lambda x: np.isfinite(x).sum()),
    )
    .reset_index()
)

approach_out = output_dir / "crossing_widths_baseline.csv"
agg.to_csv(approach_out, index=False)

print("\nSaved approach-level baseline:")
print(approach_out)

print("\nSummary:")
print(agg["crossing_width_ft"].describe())