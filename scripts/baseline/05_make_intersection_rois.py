import geopandas as gpd
from pathlib import Path

# paths
intersections_path = "../../data/vector/intersections_clean.gpkg"
output_dir = Path("../../data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# read intersections
gdf = gpd.read_file(intersections_path)

print("Loaded intersections:", len(gdf))
print("CRS:", gdf.crs)

# buffer distance in feet (EPSG:2272 is in US feet)
# start with 80 ft as a pilot radius
buffer_ft = 80

rois = gdf.copy()
rois["roi_radius_ft"] = buffer_ft
rois["geometry"] = rois.geometry.buffer(buffer_ft)

out_path = output_dir / "intersection_rois.gpkg"
rois.to_file(out_path, driver="GPKG")

print("\nSaved:")
print(out_path)
print("\nROI count:", len(rois))