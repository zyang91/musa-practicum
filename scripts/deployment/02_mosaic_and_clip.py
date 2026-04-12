"""
02_mosaic_and_clip.py
Mosaic downloaded tiles and clip to the neighborhood boundary.

Reads raw tiles from  data/raw_tiles/<neighborhood>/
Outputs clipped mosaic to  data/processed/<neighborhood>_mosaic.tif
"""

from pathlib import Path
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rio_mask
from shapely.ops import unary_union

# ── Config ──────────────────────────────────────────────────────────────────
TARGET_CRS = "EPSG:2272"
RAW_TILES_ROOT = Path("data/raw_tiles")
PROCESSED_DIR = Path("data/processed")


def main():
    parser = argparse.ArgumentParser(description="Mosaic tiles and clip to neighborhood boundary")
    parser.add_argument("--shp", required=True, help="Path to neighborhood shapefile")
    parser.add_argument("--name", required=True, help="Neighborhood name")
    args = parser.parse_args()

    shp_path = Path(args.shp)
    name = args.name
    tile_dir = RAW_TILES_ROOT / name
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_mosaic = PROCESSED_DIR / f"{name}_mosaic.tif"

    print(f"Neighborhood : {name}")
    print(f"Tile dir     : {tile_dir}")
    print(f"Output       : {output_mosaic}")

    # ── 1. List tiles ───────────────────────────────────────────────────────
    tif_files = sorted(tile_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {tile_dir}")
    print(f"Found {len(tif_files)} tiles")

    # ── 2. Mosaic ───────────────────────────────────────────────────────────
    src_files = [rasterio.open(fp) for fp in tif_files]
    try:
        mosaic_arr, mosaic_transform = merge(src_files)
    finally:
        for s in src_files:
            s.close()

    mosaic_profile = src_files[0].profile.copy()
    mosaic_profile.update(
        driver="GTiff",
        height=mosaic_arr.shape[1],
        width=mosaic_arr.shape[2],
        transform=mosaic_transform,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )

    # Write temporary mosaic (unclipped)
    tmp_mosaic = PROCESSED_DIR / f"{name}_mosaic_tmp.tif"
    with rasterio.open(tmp_mosaic, "w", **mosaic_profile) as dst:
        dst.write(mosaic_arr)
    print(f"Mosaic size  : {mosaic_arr.shape[2]} x {mosaic_arr.shape[1]} px")

    # ── 3. Clip to neighborhood boundary ────────────────────────────────────
    gdf = gpd.read_file(shp_path).to_crs(TARGET_CRS)
    geom = [unary_union(gdf.geometry)]

    with rasterio.open(tmp_mosaic) as src:
        clipped, clipped_transform = rio_mask(src, geom, crop=True)
        clip_profile = src.profile.copy()
        clip_profile.update(
            height=clipped.shape[1],
            width=clipped.shape[2],
            transform=clipped_transform,
            compress="lzw",
            BIGTIFF="IF_SAFER",
        )

    with rasterio.open(output_mosaic, "w", **clip_profile) as dst:
        dst.write(clipped)

    # Clean up temp file
    tmp_mosaic.unlink()

    print(f"Clipped size : {clipped.shape[2]} x {clipped.shape[1]} px")
    print(f"Saved        : {output_mosaic}")
    print("Done.")


if __name__ == "__main__":
    main()
