import math
import geopandas as gpd
from shapely.geometry import LineString, Point
from pathlib import Path

# paths
approaches_path = "data/vector/approach_segments.gpkg"
output_dir = Path("data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# parameters
n_transects = 5              # number of transects per approach
start_offset_ft = 20         # first transect starts 20 ft from intersection
spacing_ft = 10              # spacing between transects
half_transect_len_ft = 80    # transect extends 80 ft on each side

# read
apps = gpd.read_file(approaches_path)

print("Loaded approach segments:", len(apps))

def interpolate_point_on_line(line, dist_ft):
    if dist_ft <= 0:
        return Point(line.coords[0])
    if dist_ft >= line.length:
        return Point(line.coords[-1])
    return line.interpolate(dist_ft)

def make_perpendicular_transect(line, dist_ft, half_len_ft):
    coords = list(line.coords)
    if len(coords) < 2:
        return None

    # point along approach line
    center = interpolate_point_on_line(line, dist_ft)

    # get local direction from line start to line end
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    dx = x2 - x1
    dy = y2 - y1
    seg_len = math.hypot(dx, dy)

    if seg_len == 0:
        return None

    # unit perpendicular vector
    ux = -dy / seg_len
    uy = dx / seg_len

    x0, y0 = center.x, center.y

    p_left = (x0 - half_len_ft * ux, y0 - half_len_ft * uy)
    p_right = (x0 + half_len_ft * ux, y0 + half_len_ft * uy)

    return LineString([p_left, p_right])

records = []

for _, row in apps.iterrows():
    line = row.geometry

    for i in range(n_transects):
        dist_ft = start_offset_ft + i * spacing_ft

        if dist_ft >= line.length:
            continue

        transect = make_perpendicular_transect(
            line=line,
            dist_ft=dist_ft,
            half_len_ft=half_transect_len_ft
        )

        if transect is None:
            continue

        rec = row.copy()
        rec["transect_id"] = f"{row['approach_id']}_T{i+1}"
        rec["transect_no"] = i + 1
        rec["offset_ft"] = dist_ft
        rec["geometry"] = transect
        records.append(rec)

transects = gpd.GeoDataFrame(records, geometry="geometry", crs=apps.crs)

out_path = output_dir / "crossing_transects.gpkg"
transects.to_file(out_path, driver="GPKG")

print("\nSaved:")
print(out_path)
print("\nTransect count:", len(transects))
print("Expected approx:", len(apps) * n_transects)