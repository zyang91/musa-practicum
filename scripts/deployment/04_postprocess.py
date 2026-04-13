"""
04_postprocess.py
Clean binary prediction raster and vectorize to polygons.

Usage:
    python scripts/deployment/04_postprocess.py --input data/processed/middle_bin.tif
    python scripts/deployment/04_postprocess.py --input data/processed/south_bin.tif
    python scripts/deployment/04_postprocess.py --input data/processed/upper-middle_bin.tif
"""

from pathlib import Path
import argparse
import numpy as np
import rasterio
from rasterio.features import shapes
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import shape

# ── Config ──────────────────────────────────────────────────────────────────
MIN_PIXELS = 200        # minimum connected-component size
APPLY_OPENING = False
APPLY_CLOSING = True
STRUCTURE_SIZE = 3


def remove_small_components(mask: np.ndarray, min_pixels: int = 200) -> np.ndarray:
    labeled, num = ndimage.label(mask)
    if num == 0:
        return mask.astype(np.uint8)
    counts = np.bincount(labeled.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labeled].astype(np.uint8)


def morph_clean(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((STRUCTURE_SIZE, STRUCTURE_SIZE), dtype=np.uint8)
    out = mask.astype(bool)
    if APPLY_OPENING:
        out = ndimage.binary_opening(out, structure=structure)
    if APPLY_CLOSING:
        out = ndimage.binary_closing(out, structure=structure)
    return out.astype(np.uint8)


def raster_to_polygons(mask: np.ndarray, transform):
    geoms, vals = [], []
    for geom, value in shapes(mask, mask=(mask == 1), transform=transform):
        if value == 1:
            geoms.append(shape(geom))
            vals.append(1)
    return geoms, vals


def main():
    parser = argparse.ArgumentParser(description="Post-process predictions and vectorize")
    parser.add_argument("--input", required=True, help="Path to binary prediction raster (_bin.tif)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same folder as input)")
    parser.add_argument("--min-pixels", type=int, default=MIN_PIXELS,
                        help=f"Min component size in pixels (default: {MIN_PIXELS})")
    args = parser.parse_args()

    input_bin = Path(args.input)
    # Derive stem: e.g. "middle_bin" → "middle"
    stem = input_bin.stem.replace("_bin", "")
    out_dir = Path(args.output_dir) if args.output_dir else input_bin.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    cleaned_raster = out_dir / f"{stem}_bin_cleaned.tif"
    polygons_gpkg = out_dir / f"{stem}_crosswalks.gpkg"
    polygons_shp = out_dir / f"{stem}_crosswalks.shp"

    print(f"Input        : {input_bin}")
    print(f"Output dir   : {out_dir}")

    if not input_bin.exists():
        raise FileNotFoundError(f"Binary raster not found: {input_bin}")

    # ── Read ────────────────────────────────────────────────────────────────
    with rasterio.open(input_bin) as src:
        mask = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    mask = (mask > 0).astype(np.uint8)
    print(f"Original positive pixels: {int(mask.sum())}")

    # ── Clean ───────────────────────────────────────────────────────────────
    cleaned = remove_small_components(mask, min_pixels=args.min_pixels)
    cleaned = morph_clean(cleaned)
    print(f"Cleaned positive pixels : {int(cleaned.sum())}")

    # ── Save cleaned raster ─────────────────────────────────────────────────
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
        dst.write(cleaned, 1)
    print(f"Saved cleaned raster: {cleaned_raster}")

    # ── Vectorize ───────────────────────────────────────────────────────────
    geoms, vals = raster_to_polygons(cleaned, transform)
    if not geoms:
        print("No polygons found.")
        return

    gdf = gpd.GeoDataFrame({"value": vals}, geometry=geoms, crs=crs)
    gdf["area_sqft"] = gdf.geometry.area
    gdf = gdf[gdf.geometry.notnull() & (gdf["area_sqft"] > 0)].copy()
    gdf = gdf.reset_index(drop=True)

    print(f"Polygon count: {len(gdf)}")

    gdf.to_file(polygons_gpkg, driver="GPKG")
    gdf.to_file(polygons_shp)

    print(f"Saved: {polygons_gpkg}")
    print(f"Saved: {polygons_shp}")
    print("Done.")


if __name__ == "__main__":
    main()
