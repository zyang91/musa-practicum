import geopandas as gpd
from pathlib import Path

# paths
nodes_path = "../../data/vector/osm_nodes.gpkg"
output_dir = Path("../../data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# read nodes
nodes = gpd.read_file(nodes_path)

print("Loaded nodes:", len(nodes))
print("Columns:")
print(list(nodes.columns))

if "street_count" not in nodes.columns:
    raise ValueError("street_count column not found in osm_nodes.gpkg")

# keep likely intersections
intersections = nodes[nodes["street_count"] >= 3].copy()

# assign id
intersections = intersections.reset_index(drop=True)
intersections["intersection_id"] = [
    f"INT_{i:04d}" for i in range(1, len(intersections) + 1)
]

# save
out_path = output_dir / "intersections_clean.gpkg"
intersections.to_file(out_path, driver="GPKG")

print("\nSaved:")
print(out_path)
print("\nIntersection count:", len(intersections))
print("\nStreet-count summary:")
print(intersections["street_count"].describe())