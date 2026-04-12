"""
01_download_imagery.py
Download PASDA Philadelphia 2024 aerial imagery tiles for a neighborhood.

Source: PASDA dataset 7031 — Philadelphia 2024 orthorectified aerial imagery
        https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7031

The script uses the PASDA ArcGIS MapServer /export endpoint to download
georeferenced tiles covering the neighborhood bounding box.  The server
returns PNG images which are converted to GeoTIFF with proper CRS and
transform metadata.

Tiles are saved to  data/raw_tiles/<neighborhood>/
"""

from pathlib import Path
import argparse
import io
import math
import requests
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
import geopandas as gpd

# ── Config ──────────────────────────────────────────────────────────────────
# PASDA Philadelphia 2024 MapServer export endpoint
MAP_SERVER_URL = (
    "https://imagery.pasda.psu.edu/arcgis/rest/services/"
    "pasda/PhiladelphiaImagery2024/MapServer/export"
)

# Target CRS for output (must match training data)
TARGET_CRS = "EPSG:2272"          # PA State Plane South, US feet
TARGET_EPSG = 2272

# Tile size in pixels and spatial resolution (feet per pixel)
TILE_PX = 2048
RESOLUTION = 0.5                  # ~0.5 US feet ≈ 15 cm

# Max server export size (PASDA allows up to 4096x4096)
MAX_SERVER_PX = 4096

# Output root
RAW_TILES_ROOT = Path("data/raw_tiles")


def get_neighborhood_bbox(shp_path: Path) -> tuple:
    """Read shapefile, reproject to TARGET_CRS, return (minx, miny, maxx, maxy)."""
    gdf = gpd.read_file(shp_path)
    gdf = gdf.to_crs(TARGET_CRS)
    return tuple(gdf.total_bounds)


def download_tile(bbox, out_path: Path, size_px: int = TILE_PX) -> bool:
    """
    Download a single tile from the PASDA MapServer /export endpoint.

    The server returns a PNG image. We convert it to a georeferenced GeoTIFF
    using the known bounding box and CRS.
    """
    minx, miny, maxx, maxy = bbox

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": str(TARGET_EPSG),
        "imageSR": str(TARGET_EPSG),
        "size": f"{size_px},{size_px}",
        "format": "png",
        "transparent": "false",
        "dpi": 96,
        "layers": "show:0,1,2,3",
        "f": "image",
    }

    try:
        resp = requests.get(MAP_SERVER_URL, params=params, timeout=180)
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Request error for {out_path.name}: {e}")
        return False

    if resp.status_code != 200:
        print(f"  ✗ Failed {out_path.name} (HTTP {resp.status_code})")
        return False

    # Check that we got image data (not a JSON error)
    content_type = resp.headers.get("Content-Type", "")
    if "json" in content_type or len(resp.content) < 1000:
        print(f"  ✗ No image data for {out_path.name}")
        return False

    # Decode PNG to numpy array
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    arr = np.array(img)  # (H, W, 3)

    # Write as GeoTIFF with proper georeferencing
    height, width = arr.shape[:2]
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": width,
        "height": height,
        "count": 3,
        "crs": TARGET_CRS,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for band in range(3):
            dst.write(arr[:, :, band], band + 1)

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
    parser = argparse.ArgumentParser(
        description="Download PASDA imagery tiles for a neighborhood"
    )
    parser.add_argument("--shp", required=True,
                        help="Path to neighborhood shapefile (.shp)")
    parser.add_argument("--name", required=True,
                        help="Neighborhood name (used for output folder)")
    parser.add_argument("--buffer", type=float, default=200.0,
                        help="Buffer around boundary in feet (default: 200)")
    parser.add_argument("--tile-px", type=int, default=TILE_PX,
                        help=f"Tile size in pixels (default: {TILE_PX})")
    parser.add_argument("--resolution", type=float, default=RESOLUTION,
                        help=f"Spatial resolution in ft/px (default: {RESOLUTION})")
    args = parser.parse_args()

    shp_path = Path(args.shp)
    name = args.name
    tile_px = args.tile_px
    resolution = args.resolution
    out_dir = RAW_TILES_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Neighborhood : {name}")
    print(f"Shapefile    : {shp_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Server URL   : {MAP_SERVER_URL}")

    # Get bounding box with buffer
    minx, miny, maxx, maxy = get_neighborhood_bbox(shp_path)
    minx -= args.buffer
    miny -= args.buffer
    maxx += args.buffer
    maxy += args.buffer
    print(f"Bounding box : ({minx:.1f}, {miny:.1f}, {maxx:.1f}, {maxy:.1f})")

    # Compute tile grid
    tile_size_ft = tile_px * resolution
    tiles, n_cols, n_rows = compute_tile_grid(
        (minx, miny, maxx, maxy), tile_size_ft
    )
    print(f"Tile grid    : {n_cols} x {n_rows} = {len(tiles)} tiles")
    print(f"Tile size    : {tile_px}px @ {resolution} ft/px = {tile_size_ft:.0f} ft")

    # Download tiles
    success = 0
    for i, tile_bbox in enumerate(tiles):
        tile_path = out_dir / f"tile_{i:04d}.tif"
        if tile_path.exists():
            print(f"  ✓ Tile {i+1}/{len(tiles)} already exists, skipping")
            success += 1
            continue

        print(f"  Downloading tile {i+1}/{len(tiles)} ...", end=" ", flush=True)
        if download_tile(tile_bbox, tile_path, size_px=tile_px):
            print("✓")
            success += 1
        else:
            print("✗")

    print(f"\nDownloaded {success}/{len(tiles)} tiles to {out_dir}")
    if success < len(tiles):
        print(f"WARNING: {len(tiles) - success} tiles failed. Re-run to retry.")


if __name__ == "__main__":
    main()
