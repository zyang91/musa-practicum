import math
import geopandas as gpd
from shapely.geometry import LineString, Point
from pathlib import Path

# paths
approaches_path = "data/vector/approaches_raw.gpkg"
intersections_path = "data/vector/intersections_inner.gpkg"
output_dir = Path("data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# parameters
segment_length_ft = 120  # keep first 120 ft from the intersection

# read data
apps = gpd.read_file(approaches_path)
ints = gpd.read_file(intersections_path)

print("Loaded approaches:", len(apps))
print("Loaded intersections:", len(ints))

# build lookup for intersection point
int_lookup = ints.set_index("intersection_id")["geometry"].to_dict()

def cut_line_from_point(line, start_point, dist_ft):
    """
    Return a short segment of 'line' starting from the line end nearest to start_point,
    with length approximately dist_ft.
    """
    if line is None or line.is_empty:
        return None

    coords = list(line.coords)
    if len(coords) < 2:
        return None

    p0 = Point(coords[0])
    p1 = Point(coords[-1])

    # decide which end is the intersection end
    if start_point.distance(p0) <= start_point.distance(p1):
        ordered = coords
    else:
        ordered = list(reversed(coords))

    new_coords = [ordered[0]]
    cumdist = 0.0

    for i in range(len(ordered) - 1):
        a = Point(ordered[i])
        b = Point(ordered[i + 1])
        seg_len = a.distance(b)

        if cumdist + seg_len <= dist_ft:
            new_coords.append(ordered[i + 1])
            cumdist += seg_len
        else:
            remain = dist_ft - cumdist
            if seg_len > 0 and remain > 0:
                ratio = remain / seg_len
                x = a.x + ratio * (b.x - a.x)
                y = a.y + ratio * (b.y - a.y)
                new_coords.append((x, y))
            break

    if len(new_coords) < 2:
        return None

    return LineString(new_coords)

def compute_bearing_deg(line):
    """
    Bearing of the short approach segment from intersection outward.
    0 = east, 90 = north (math-style angle converted to degrees)
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return None

    x1, y1 = coords[0]
    x2, y2 = coords[-1]

    dx = x2 - x1
    dy = y2 - y1

    angle = math.degrees(math.atan2(dy, dx))
    if angle < 0:
        angle += 360

    return angle

records = []

for _, row in apps.iterrows():
    intersection_id = row["intersection_id"]
    line = row.geometry

    if intersection_id not in int_lookup:
        continue

    int_pt = int_lookup[intersection_id]

    short_seg = cut_line_from_point(line, int_pt, segment_length_ft)
    if short_seg is None:
        continue

    bearing_deg = compute_bearing_deg(short_seg)

    rec = row.copy()
    rec["geometry"] = short_seg
    rec["segment_len_ft"] = short_seg.length
    rec["bearing_deg"] = bearing_deg
    records.append(rec)

approach_segments = gpd.GeoDataFrame(records, geometry="geometry", crs=apps.crs)

# save
out_path = output_dir / "approach_segments.gpkg"
approach_segments.to_file(out_path, driver="GPKG")

print("\nSaved:")
print(out_path)
print("\nApproach segment count:", len(approach_segments))
print("\nExample columns:")
print([c for c in approach_segments.columns if c in ["approach_id", "intersection_id", "node_role", "segment_len_ft", "bearing_deg"]])