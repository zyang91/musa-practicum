from pathlib import Path
import rasterio
import numpy as np

mask_path = Path("data/seg/road_surface_mask.tif")

with rasterio.open(mask_path) as src:
    arr = src.read(1)
    print("Mask shape:", arr.shape)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print("Unique values:", np.unique(arr))
    print("Positive pixels:", int((arr > 0).sum()))