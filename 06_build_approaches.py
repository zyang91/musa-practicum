import geopandas as gpd
import pandas as pd
from pathlib import Path

# paths
intersections_path = "data/vector/intersections_inner.gpkg"
edges_path = "data/vector/osm_edges.gpkg"
output_dir = Path("data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# read data
ints = gpd.read_file(intersections_path)
edges = gpd.read_file(edges_path)

print("Loaded intersections:", len(ints))
print("Loaded edges:", len(edges))

# keep only needed columns from intersections
ints2 = ints[["intersection_id", "osmid", "geometry"]].copy()
ints2["osmid"] = ints2["osmid"].astype(str)

# edge endpoints
edges2 = edges.copy()
edges2["u"] = edges2["u"].astype(str)
edges2["v"] = edges2["v"].astype(str)

# create approach records for edges touching an intersection at u
app_u = edges2.merge(
    ints2[["intersection_id", "osmid"]],
    left_on="u",
    right_on="osmid",
    how="inner"
).copy()
app_u["node_role"] = "u"

# create approach records for edges touching an intersection at v
app_v = edges2.merge(
    ints2[["intersection_id", "osmid"]],
    left_on="v",
    right_on="osmid",
    how="inner"
).copy()
app_v["node_role"] = "v"

# combine
approaches = pd.concat([app_u, app_v], ignore_index=True)

# remove exact duplicates
dedup_cols = ["intersection_id", "u", "v", "key"]
dedup_cols = [c for c in dedup_cols if c in approaches.columns]
approaches = approaches.drop_duplicates(subset=dedup_cols)

# assign approach ids
approaches = approaches.reset_index(drop=True)
approaches["approach_id"] = [f"APP_{i:05d}" for i in range(1, len(approaches) + 1)]

# convert back to GeoDataFrame
approaches = gpd.GeoDataFrame(approaches, geometry="geometry", crs=edges.crs)

# save
out_path = output_dir / "approaches_raw.gpkg"
approaches.to_file(out_path, driver="GPKG")

print("\nSaved:")
print(out_path)
print("\nApproach count:", len(approaches))

print("\nColumns:")
print(list(approaches.columns))