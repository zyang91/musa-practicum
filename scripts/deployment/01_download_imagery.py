"""
01_download_imagery.py
Print download instructions for PASDA Philadelphia 2024 aerial imagery.

Manual download options:
  1. PASDA Imagery Navigator: http://maps.psiee.psu.edu/imagerynavigator/
  2. PASDA dataset page:      https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7031

After downloading, place .tif tiles in:
    data/raw_tiles/<neighborhood>/

Usage:
    python scripts/deployment/01_download_imagery.py
    python scripts/deployment/01_download_imagery.py --name KENSINGTON
"""

from pathlib import Path
import argparse
import geopandas as gpd

TARGET_CRS = "EPSG:2272"   # PA State Plane South, US feet
SHP_DIR = Path("shp")
RAW_TILES_ROOT = Path("data/raw_tiles")

NEIGHBORHOODS = {
    "Center_City": "Center City",
    "KENSINGTON": "KENSINGTON",
    "POINT_BREEZE": "POINT_BREEZE",
}


def print_info(name: str, shp_stem: str):
    shp_path = SHP_DIR / f"{shp_stem}.shp"
    gdf = gpd.read_file(shp_path)

    # Bounds in original CRS (Web Mercator → lat/lon for reference)
    gdf_4326 = gdf.to_crs("EPSG:4326")
    lon_min, lat_min, lon_max, lat_max = gdf_4326.total_bounds

    # Bounds in PA State Plane (the imagery CRS)
    gdf_2272 = gdf.to_crs(TARGET_CRS)
    xmin, ymin, xmax, ymax = gdf_2272.total_bounds

    tile_dir = RAW_TILES_ROOT / name
    existing = list(tile_dir.glob("*.tif")) if tile_dir.exists() else []

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Shapefile     : {shp_path}")
    print(f"  Lat/Lon bbox  : ({lat_min:.5f}, {lon_min:.5f}) to ({lat_max:.5f}, {lon_max:.5f})")
    print(f"  EPSG:2272 bbox: ({xmin:.0f}, {ymin:.0f}) to ({xmax:.0f}, {ymax:.0f})")
    print(f"  Tile folder   : {tile_dir}/")
    print(f"  Existing tiles: {len(existing)}")
    if existing:
        print(f"    ✓ Ready for mosaic")
    else:
        print(f"    ✗ No tiles yet — download and place .tif files here")


def main():
    parser = argparse.ArgumentParser(description="Print imagery download instructions")
    parser.add_argument("--name", default=None, help="Specific neighborhood (default: show all)")
    args = parser.parse_args()

    print("PASDA Philadelphia 2024 Aerial Imagery — Manual Download Guide")
    print()
    print("Download sources:")
    print("  1. PASDA Imagery Navigator : http://maps.psiee.psu.edu/imagerynavigator/")
    print("  2. PASDA dataset page      : https://www.pasda.psu.edu/uci/DataSummary.aspx?dataset=7031")
    print()
    print("Instructions:")
    print("  - Select Philadelphia 2024 imagery layer")
    print("  - Navigate to the neighborhood area using the bbox coordinates below")
    print("  - Download tiles covering the full neighborhood extent")
    print(f"  - Place downloaded .tif files in: data/raw_tiles/<neighborhood>/")
    print(f"  - Imagery CRS should be EPSG:2272 (PA State Plane South, US feet)")

    if args.name:
        targets = {k: v for k, v in NEIGHBORHOODS.items() if k == args.name}
    else:
        targets = NEIGHBORHOODS

    for name, shp_stem in targets.items():
        print_info(name, shp_stem)

    print(f"\n{'='*60}")
    print("After placing tiles, run the pipeline with --skip-download:")
    print("  python scripts/deployment/run_pipeline.py --skip-download")
    print()


if __name__ == "__main__":
    main()
