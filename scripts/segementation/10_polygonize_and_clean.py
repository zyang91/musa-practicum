from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import shape


# =========================
# Config
# =========================
INPUT_BIN = Path("outputs/full_scene/university_city_pilot_mosaic_bin.tif")
OUTPUT_DIR = Path("outputs/vectorized")

# minimum connected-component size in pixels
MIN_PIXELS = 200

# optional morphological cleanup
APPLY_OPENING = False
APPLY_CLOSING = True

# structure size for morphology
STRUCTURE_SIZE = 3


def remove_small_components(mask, min_pixels=200):
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask.astype(np.uint8)

    counts = np.bincount(labeled.ravel())
    keep = counts >= min_pixels
    keep[0] = False  # background

    cleaned = keep[labeled]
    return cleaned.astype(np.uint8)


def morph_clean(mask, apply_opening=False, apply_closing=True, structure_size=3):
    structure = np.ones((structure_size, structure_size), dtype=np.uint8)

    out = mask.astype(bool)

    if apply_opening:
        out = ndimage.binary_opening(out, structure=structure)

    if apply_closing:
        out = ndimage.binary_closing(out, structure=structure)

    return out.astype(np.uint8)


def raster_to_polygons(mask, transform):
    geoms = []
    vals = []

    for geom, value in shapes(mask, mask=(mask == 1), transform=transform):
        if value == 1:
            geoms.append(shape(geom))
            vals.append(1)

    return geoms, vals


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_raster = OUTPUT_DIR / f"{INPUT_BIN.stem}_cleaned.tif"
    polygons_gpkg = OUTPUT_DIR / f"{INPUT_BIN.stem}_polygons.gpkg"
    polygons_shp = OUTPUT_DIR / f"{INPUT_BIN.stem}_polygons.shp"

    with rasterio.open(INPUT_BIN) as src:
        mask = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    mask = (mask > 0).astype(np.uint8)
    print("Original positive pixels:", int(mask.sum()))

    cleaned = remove_small_components(mask, min_pixels=MIN_PIXELS)
    cleaned = morph_clean(
        cleaned,
        apply_opening=APPLY_OPENING,
        apply_closing=APPLY_CLOSING,
        structure_size=STRUCTURE_SIZE
    )

    print("Cleaned positive pixels:", int(cleaned.sum()))

    out_profile = {
        "driver": "GTiff",
        "height": cleaned.shape[0],
        "width": cleaned.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "BIGTIFF": "YES",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }

    with rasterio.open(cleaned_raster, "w", **out_profile) as dst:
        dst.write(cleaned.astype(np.uint8), 1)

    print("Saved cleaned raster:", cleaned_raster)

    geoms, vals = raster_to_polygons(cleaned, transform)

    if len(geoms) == 0:
        print("No polygons found.")
        return

    gdf = gpd.GeoDataFrame(
        {"value": vals},
        geometry=geoms,
        crs=crs
    )

    gdf["area"] = gdf.geometry.area
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.area > 0].copy()

    print("Polygon count:", len(gdf))

    gdf.to_file(polygons_gpkg, driver="GPKG")
    gdf.to_file(polygons_shp)

    print("Saved polygons:", polygons_gpkg)
    print("Saved polygons:", polygons_shp)
    print("Done.")


if __name__ == "__main__":
    main()