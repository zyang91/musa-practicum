from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import unary_union


# =========================
# Config
# =========================
INTERSECTIONS_PATH = Path("data/vector/intersections_inner.gpkg")
EDGES_PATH = Path("data/vector/osm_edges.gpkg")
ROAD_POLYGONS_PATH = Path("outputs/vectorized/university_city_pilot_mosaic_bin_polygons.gpkg")
OUTPUT_DIR = Path("outputs/crossing_distance")

# geometry parameters, in feet (EPSG:2272)
SEARCH_BUFFER_FT = 20          # to catch edges near intersection point
DIRECTION_SAMPLE_FT = 30       # estimate road direction from first ~30 ft
CROSSING_OFFSET_FT = 35        # move outward from node before measuring crossing
MEASURE_HALF_LEN_FT = 120      # total measure line length = 240 ft
MIN_VALID_DISTANCE_FT = 5
MAX_VALID_DISTANCE_FT = 200


# =========================
# Helpers
# =========================
def unit_vector(dx, dy):
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return dx / norm, dy / norm


def point_along_from_intersection(line, intersection_pt, sample_ft=30):
    """
    Returns a point on the line near the intersection, moving away from the intersection.
    """
    if line.is_empty or line.length == 0:
        return None

    coords = list(line.coords)
    start = Point(coords[0])
    end = Point(coords[-1])

    d_start = intersection_pt.distance(start)
    d_end = intersection_pt.distance(end)

    # choose the nearer endpoint as the intersection-side endpoint
    if d_start <= d_end:
        anchor_dist = 0
    else:
        anchor_dist = line.length

    if anchor_dist == 0:
        dist2 = min(sample_ft, line.length)
        return line.interpolate(dist2)
    else:
        dist2 = max(line.length - sample_ft, 0)
        return line.interpolate(dist2)


def local_direction_vector(line, intersection_pt, sample_ft=30):
    """
    Estimate edge direction away from intersection.
    """
    p2 = point_along_from_intersection(line, intersection_pt, sample_ft)
    if p2 is None:
        return None

    dx = p2.x - intersection_pt.x
    dy = p2.y - intersection_pt.y
    uv = unit_vector(dx, dy)
    return uv


def make_measure_line(intersection_pt, dir_uv, offset_ft=35, half_len_ft=120):
    """
    Build a crossing-measure line:
    - move from intersection point outward along road direction
    - draw a perpendicular line centered there
    """
    ux, uy = dir_uv

    # outward point along road direction
    cx = intersection_pt.x + ux * offset_ft
    cy = intersection_pt.y + uy * offset_ft

    # perpendicular direction
    px, py = -uy, ux

    p1 = Point(cx - px * half_len_ft, cy - py * half_len_ft)
    p2 = Point(cx + px * half_len_ft, cy + py * half_len_ft)

    return LineString([p1, p2]), Point(cx, cy)


def length_of_line_inside_polygon(measure_line, road_union, center_pt):
    """
    Intersect measure line with road polygon and return the segment length
    corresponding to the road width around center_pt.
    """
    inter = measure_line.intersection(road_union)

    if inter.is_empty:
        return None

    candidate_lines = []

    if inter.geom_type == "LineString":
        candidate_lines = [inter]
    elif inter.geom_type == "MultiLineString":
        candidate_lines = list(inter.geoms)
    elif inter.geom_type == "GeometryCollection":
        candidate_lines = [g for g in inter.geoms if g.geom_type == "LineString"]

    if not candidate_lines:
        return None

    # choose the segment nearest the crossing center point
    best = min(candidate_lines, key=lambda g: g.distance(center_pt))
    return best.length, best


def get_intersection_id_col(gdf):
    if "intersection_id" in gdf.columns:
        return "intersection_id"
    return None


def get_node_id_col(gdf):
    for c in ["osmid", "osmid_original", "node_id"]:
        if c in gdf.columns:
            return c
    return None


# =========================
# Main
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_points = OUTPUT_DIR / "crossing_distance_points.gpkg"
    out_lines = OUTPUT_DIR / "crossing_distance_lines.gpkg"
    out_csv = OUTPUT_DIR / "crossing_distance_table.csv"

    intersections = gpd.read_file(INTERSECTIONS_PATH)
    edges = gpd.read_file(EDGES_PATH)
    road_polys = gpd.read_file(ROAD_POLYGONS_PATH)

    # harmonize CRS
    edges = edges.to_crs(intersections.crs)
    road_polys = road_polys.to_crs(intersections.crs)

    print("Loaded intersections:", len(intersections))
    print("Loaded edges:", len(edges))
    print("Loaded road polygons:", len(road_polys))
    print("CRS:", intersections.crs)

    intersection_id_col = get_intersection_id_col(intersections)
    node_id_col = get_node_id_col(intersections)

    if intersection_id_col is None:
        intersections["intersection_id"] = [f"INT_{i:04d}" for i in range(1, len(intersections) + 1)]
        intersection_id_col = "intersection_id"

    # road union for fast line intersection
    road_union = unary_union(road_polys.geometry)

    records = []
    line_records = []

    # spatial index candidate search
    edges_sindex = edges.sindex

    for idx, row in intersections.iterrows():
        ipt = row.geometry
        intersection_id = row[intersection_id_col]
        node_id = row[node_id_col] if node_id_col is not None else None

        # first try by u/v node id match
        cand = None
        if node_id is not None and "u" in edges.columns and "v" in edges.columns:
            cand = edges[(edges["u"] == node_id) | (edges["v"] == node_id)].copy()
        else:
            cand = gpd.GeoDataFrame(columns=edges.columns, geometry=[], crs=edges.crs)

        # fallback/additional spatial catch
        search_geom = ipt.buffer(SEARCH_BUFFER_FT)
        possible_idx = list(edges_sindex.intersection(search_geom.bounds))
        spatial_cand = edges.iloc[possible_idx].copy()
        spatial_cand = spatial_cand[spatial_cand.geometry.intersects(search_geom)]

        if len(cand) == 0:
            cand = spatial_cand.copy()
        else:
            cand = pd.concat([cand, spatial_cand], ignore_index=True).drop_duplicates()

        if len(cand) == 0:
            continue

        leg_num = 0
        used_dirs = []

        for _, erow in cand.iterrows():
            geom = erow.geometry
            if geom is None or geom.is_empty:
                continue

            if geom.geom_type == "MultiLineString":
                parts = list(geom.geoms)
            else:
                parts = [geom]

            for part in parts:
                dir_uv = local_direction_vector(part, ipt, DIRECTION_SAMPLE_FT)
                if dir_uv is None:
                    continue

                ux, uy = dir_uv
                angle = math.degrees(math.atan2(uy, ux)) % 180

                # simple de-dup by direction
                duplicate = False
                for a in used_dirs:
                    diff = abs(angle - a)
                    diff = min(diff, 180 - diff)
                    if diff < 15:
                        duplicate = True
                        break
                if duplicate:
                    continue

                used_dirs.append(angle)
                leg_num += 1

                measure_line, center_pt = make_measure_line(
                    ipt,
                    dir_uv,
                    offset_ft=CROSSING_OFFSET_FT,
                    half_len_ft=MEASURE_HALF_LEN_FT,
                )

                result = length_of_line_inside_polygon(measure_line, road_union, center_pt)
                if result is None:
                    continue

                crossing_ft, crossing_seg = result

                if crossing_ft < MIN_VALID_DISTANCE_FT or crossing_ft > MAX_VALID_DISTANCE_FT:
                    continue

                records.append({
                    "intersection_id": intersection_id,
                    "leg_id": f"{intersection_id}_LEG_{leg_num:02d}",
                    "angle_deg": angle,
                    "crossing_ft": float(crossing_ft),
                    "geometry": center_pt
                })

                line_records.append({
                    "intersection_id": intersection_id,
                    "leg_id": f"{intersection_id}_LEG_{leg_num:02d}",
                    "angle_deg": angle,
                    "crossing_ft": float(crossing_ft),
                    "geometry": crossing_seg
                })

    if not records:
        print("No crossing distances were generated.")
        return

    points_gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=intersections.crs)
    lines_gdf = gpd.GeoDataFrame(line_records, geometry="geometry", crs=intersections.crs)

    # summary by intersection
    summary = (
        points_gdf.groupby("intersection_id", as_index=False)
        .agg(
            n_legs=("leg_id", "count"),
            mean_crossing_ft=("crossing_ft", "mean"),
            max_crossing_ft=("crossing_ft", "max"),
            min_crossing_ft=("crossing_ft", "min"),
        )
    )

    points_gdf.to_file(out_points, driver="GPKG")
    lines_gdf.to_file(out_lines, driver="GPKG")
    summary.to_csv(out_csv, index=False)

    print("\nSaved:")
    print(out_points)
    print(out_lines)
    print(out_csv)

    print("\nCrossing distance summary (ft):")
    print(points_gdf["crossing_ft"].describe())

    print("\nDone.")


if __name__ == "__main__":
    main()