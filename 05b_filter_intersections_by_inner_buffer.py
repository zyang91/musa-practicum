import geopandas as gpd
import rasterio
from shapely.geometry import box
from pathlib import Path

# paths
mosaic_path = "data/processed/university_city_pilot_mosaic.tif"
intersections_path = "data/vector/intersections_clean.gpkg"
output_dir = Path("data/vector")
output_dir.mkdir(parents=True, exist_ok=True)

# read mosaic bounds
with rasterio.open(mosaic_path) as src:
    bounds = src.bounds
    crs = src.crs

# make pilot boundary polygon
pilot_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
pilot_gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[pilot_poly], crs=crs)

# inward buffer to remove edge intersections
# 150 ft is a reasonable first-pass margin
inner_margin_ft = 150
pilot_inner = pilot_gdf.copy()
pilot_inner["geometry"] = pilot_inner.geometry.buffer(-inner_margin_ft)

# read intersections
ints = gpd.read_file(intersections_path).to_crs(crs)

# keep only intersections fully inside inner polygon
inner_geom = pilot_inner.geometry.iloc[0]
ints_keep = ints[ints.geometry.within(inner_geom)].copy()

out_path = output_dir / "intersections_inner.gpkg"
ints_keep.to_file(out_path, driver="GPKG")

print("Original intersections:", len(ints))
print("Filtered intersections:", len(ints_keep))
print("Saved:", out_path)