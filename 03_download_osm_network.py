import rasterio
import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

# paths
mosaic_path = "data/processed/university_city_pilot_mosaic.tif"
output_dir = Path("data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# read mosaic bounds
with rasterio.open(mosaic_path) as src:
    bounds = src.bounds
    img_crs = src.crs

print("Image CRS:", img_crs)
print("Image bounds:", bounds)

# build bounding box polygon in image CRS
bbox_geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
bbox_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[bbox_geom], crs=img_crs)

# convert to WGS84 for OSM query
bbox_wgs = bbox_gdf.to_crs(epsg=4326)
minx, miny, maxx, maxy = bbox_wgs.total_bounds

print("\nOSM query bbox (WGS84):")
print("west  =", minx)
print("south =", miny)
print("east  =", maxx)
print("north =", maxy)

# correct bbox order for osmnx 2.x:
# (left, bottom, right, top) = (west, south, east, north)
G = ox.graph_from_bbox(
    bbox=(minx, miny, maxx, maxy),
    network_type="drive",
    simplify=True
)

# convert to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(G)

# project back to image CRS
nodes = nodes.to_crs(img_crs)
edges = edges.to_crs(img_crs)

# save
nodes_path = output_dir / "osm_nodes.gpkg"
edges_path = output_dir / "osm_edges.gpkg"

nodes.to_file(nodes_path, driver="GPKG")
edges.to_file(edges_path, driver="GPKG")

print("\nSaved:")
print(nodes_path)
print(edges_path)

print("\nCounts:")
print("nodes:", len(nodes))
print("edges:", len(edges))