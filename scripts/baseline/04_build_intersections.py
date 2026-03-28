import geopandas as gpd
import pandas as pd
from pathlib import Path

# paths
nodes_path = "../../data/vector/osm_nodes.gpkg"
edges_path = "../../data/vector/osm_edges.gpkg"
output_dir = Path("../../data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# read data
nodes = gpd.read_file(nodes_path)
edges = gpd.read_file(edges_path)

print("Loaded:")
print("nodes:", len(nodes))
print("edges:", len(edges))

# OSM edges usually have u and v columns for from/to node ids
if "u" not in edges.columns or "v" not in edges.columns:
    raise ValueError("Edges file does not contain 'u' and 'v' columns.")

# count node degree from edges
u_counts = edges["u"].value_counts()
v_counts = edges["v"].value_counts()
degree = u_counts.add(v_counts, fill_value=0)

degree_df = degree.reset_index()
degree_df.columns = ["osmid", "degree"]

# join back to nodes
# nodes index may already be osmid, but we handle both cases
if "osmid" in nodes.columns:
    nodes2 = nodes.merge(degree_df, on="osmid", how="left")
else:
    nodes2 = nodes.reset_index().rename(columns={"index": "osmid"})
    nodes2 = nodes2.merge(degree_df, on="osmid", how="left")

nodes2["degree"] = nodes2["degree"].fillna(0)

# keep likely intersections
intersections = nodes2[nodes2["degree"] >= 3].copy()

# assign ids
intersections = intersections.reset_index(drop=True)
intersections["intersection_id"] = [
    f"INT_{i:04d}" for i in range(1, len(intersections) + 1)
]

# save
out_path = output_dir / "intersections.gpkg"
intersections.to_file(out_path, driver="GPKG")

print("\nSaved:")
print(out_path)
print("\nIntersection count:", len(intersections))
print("\nDegree summary:")
print(intersections["degree"].describe())
