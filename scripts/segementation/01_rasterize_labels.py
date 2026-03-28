from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np

# paths
image_path = Path("data/processed/university_city_pilot_mosaic.tif")
label_path = Path("data/label/label.shp")
output_dir = Path("data/seg")
output_dir.mkdir(parents=True, exist_ok=True)

mask_path = output_dir / "road_surface_mask.tif"

# read reference raster
with rasterio.open(image_path) as src:
    ref_meta = src.meta.copy()
    ref_transform = src.transform
    ref_crs = src.crs
    ref_shape = (src.height, src.width)

print("Reference raster:")
print("  CRS:", ref_crs)
print("  Shape:", ref_shape)

# read labels
gdf = gpd.read_file(label_path)

print("\nLoaded labels:", len(gdf))
print("Label CRS:", gdf.crs)
print("Columns:", list(gdf.columns))

# reproject labels if needed
if gdf.crs != ref_crs:
    print("\nReprojecting labels to match raster CRS...")
    gdf = gdf.to_crs(ref_crs)

# determine burn value
if "label" in gdf.columns:
    shapes = ((geom, int(val)) for geom, val in zip(gdf.geometry, gdf["label"]))
else:
    print("\n'label' field not found. Using value = 1 for all polygons.")
    shapes = ((geom, 1) for geom in gdf.geometry)

# rasterize
mask = rasterize(
    shapes=shapes,
    out_shape=ref_shape,
    transform=ref_transform,
    fill=0,
    dtype="uint8"
)

# save mask
mask_meta = ref_meta.copy()
mask_meta.update({
    "count": 1,
    "dtype": "uint8",
    "compress": "lzw"
})

with rasterio.open(mask_path, "w", **mask_meta) as dst:
    dst.write(mask, 1)

print("\nSaved mask to:", mask_path)
print("Unique values in mask:", np.unique(mask))