"""
01_download_imagery.py
Download PASDA Philadelphia 2024 aerial imagery tiles for a neighborhood.

Source: PASDA dataset 7031 — Philadelphia 2024 orthorectified aerial imagery
        https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7031

The script uses the PASDA ArcGIS ImageServer export endpoint to download
georeferenced tiles that cover the neighborhood bounding box.  Tiles are
saved as GeoTIFF files in  data/raw_tiles/<neighborhood>/
"""

from pathlib import Path
import argparse
import math
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import box

# ── Config ──────────────────────────────────────────────────────────────────
# PASDA Philadelphia 2024 ImageServer export endpoint
# Update this URL if the service location changes.
IMAGE_SERVER_URL = (
    "https://imagery.pasda.psu.edu/arcgis/rest/services/"
    "Imagery/PhiladelphiaPA2024/ImageServer/exportImage"
)

# Target CRS for the imagery (must match training data)
TARGET_CRS = "EPSG:2272"   # PA State Plane South, US feet

# Tile size in pixels and spatial resolution (feet per pixel)
TILE_PX = 2048
RESOLUTION = 0.5           # ~0.5 US feet ≈ 15 cm

# Output root
RAW_TILES_ROOT = Path("data/raw_tiles")


def get_neighborhood_bbox(shp_path: Path) -> tuple:
    """Read shapefile, reproject to TARGET_CRS, return (minx, miny, maxx, maxy)."""
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(TARGET_CRS)
    return tuple(gdf.total_bounds)  # minx, miny, maxx, maxy


def download_tile(bbox, out_path: Path, size_px: int = TILE_PX) -> bool:
    """Download a single tile from the PASDA ImageServer export endpoint."""
    minx, miny, maxx, maxy = bbox
    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": TARGET_CRS.split(":")[1],
        "imageSR": TARGET_CRS.split(":")[1],
        "size": f"{size_px},{size_px}",
        "format": "tiff",
        "pixelType": "U8",
        "noData": "",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    resp = requests.get(IMAGE_SERVER_URL, params=params, timeout=120)
    if resp.status_code != 200 or len(resp.content) < 1000:
        print(f"  ✗ Failed tile {out_path.name} (HTTP {resp.status_code})")
        return False

    out_path.write_bytes(resp.content)
    return True


def compute_tile_grid(bbox, tile_size_ft):
    """Compute a grid of tile bounding boxes covering the full bbox."""
    minx, miny, maxx, maxy = bbox
    cols = math.ceil((maxx - minx) / tile_size_ft)
    rows = math.ceil((maxy - miny) / tile_size_ft)

    tiles = []
    for row in range(rows):
        for col in range(cols):
            tx0 = minx + col * tile_size_ft
            ty0 = miny + row * tile_size_ft
            tx1 = tx0 + tile_size_ft
            ty1 = ty0 + tile_size_ft
            tiles.append((tx0, ty0, tx1, ty1))
    return tiles, cols, rows


def main():
    parser = argparse.ArgumentParser(description="Download PASDA imagery tiles for a neighborhood")
    parser.add_argument("--shp", required=True, help="Path to neighborhood shapefile (.shp)")
    parser.add_argument("--name", required=True, help="Neighborhood name (used for output folder)")
    parser.add_argument("--buffer", type=float, default=200.0,
                        help="Buffer around boundary in feet (default: 200)")
    args = parser.parse_args()

    shp_path = Path(args.shp)
    name = args.name
    out_dir = RAW_TILES_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Neighborhood : {name}")
    print(f"Shapefile    : {shp_path}")
    print(f"Output dir   : {out_dir}")

    # Get bounding box with buffer
    minx, miny, maxx, maxy = get_neighborhood_bbox(shp_path)
    minx -= args.buffer
    miny -= args.buffer
    maxx += args.buffer
    maxy += args.buffer
    print(f"Bounding box : ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")

    # Compute tile grid
    tile_size_ft = TILE_PX * RESOLUTION
    tiles, n_cols, n_rows = compute_tile_grid(
        (minx, miny, maxx, maxy), tile_size_ft
    )
    print(f"Tile grid    : {n_cols} x {n_rows} = {len(tiles)} tiles")
    print(f"Tile size    : {TILE_PX}px @ {RESOLUTION} ft/px = {tile_size_ft:.0f} ft")

    # Download tiles
    success = 0
    for i, tile_bbox in enumerate(tiles):
        tile_path = out_dir / f"tile_{i:04d}.tif"
        if tile_path.exists():
            print(f"  ✓ Tile {i+1}/{len(tiles)} already exists, skipping")
            success += 1
            continue

        print(f"  Downloading tile {i+1}/{len(tiles)} ...", end=" ")
        if download_tile(tile_bbox, tile_path):
            print("✓")
            success += 1
        else:
            print("✗")

    print(f"\nDownloaded {success}/{len(tiles)} tiles to {out_dir}")


if __name__ == "__main__":
    main()
