from pathlib import Path
import rasterio
from rasterio.merge import merge

# 1. input/output paths
input_dir = Path("data/raw_tiles")
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

output_mosaic = output_dir / "university_city_pilot_mosaic.tif"

# 2. list tif files
tif_files = sorted(input_dir.glob("*.tif"))

if len(tif_files) == 0:
    raise FileNotFoundError(f"No .tif files found in {input_dir}")

print("Found TIFF files:")
for f in tif_files:
    print(" -", f.name)

# 3. open rasters
src_files = [rasterio.open(fp) for fp in tif_files]

try:
    # 4. merge rasters
    mosaic, out_transform = merge(src_files)

    # 5. use metadata from first tile as template
    out_meta = src_files[0].meta.copy()
    out_meta.update(
        {
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
            "compress": "lzw",
            "BIGTIFF": "IF_SAFER",
        }
    )

    # 6. save mosaic
    with rasterio.open(output_mosaic, "w", **out_meta) as dest:
        dest.write(mosaic)

    print("\nSaved mosaic to:")
    print(output_mosaic)

    print("\nMosaic info:")
    print(" - bands:", mosaic.shape[0])
    print(" - height:", mosaic.shape[1])
    print(" - width:", mosaic.shape[2])
    print(" - CRS:", out_meta["crs"])

finally:
    for src in src_files:
        src.close()