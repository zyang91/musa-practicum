from pathlib import Path
import rasterio
from rasterio.windows import Window
import numpy as np
from PIL import Image

# paths
image_path = Path("data/processed/university_city_pilot_mosaic.tif")
mask_path = Path("data/seg/road_surface_mask.tif")

out_img_dir = Path("data/seg/images")
out_mask_dir = Path("data/seg/masks")
out_img_dir.mkdir(parents=True, exist_ok=True)
out_mask_dir.mkdir(parents=True, exist_ok=True)

# clear old patches
for p in out_img_dir.glob("*.png"):
    p.unlink()
for p in out_mask_dir.glob("*.png"):
    p.unlink()

# parameters
patch_size = 512
stride = 256              # overlap
min_positive_ratio = 0.002
keep_empty_ratio = 0.03   # keep fewer empty patches

rng = np.random.default_rng(42)

saved = 0
saved_positive = 0
saved_empty = 0

with rasterio.open(image_path) as img_src, rasterio.open(mask_path) as mask_src:
    assert img_src.width == mask_src.width and img_src.height == mask_src.height
    assert img_src.transform == mask_src.transform

    H, W = img_src.height, img_src.width
    patch_id = 0

    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            win = Window(left, top, patch_size, patch_size)

            img = img_src.read([1, 2, 3], window=win)
            img = np.transpose(img, (1, 2, 0))

            mask = mask_src.read(1, window=win)
            positive_ratio = (mask > 0).mean()

            keep = False
            if positive_ratio >= min_positive_ratio:
                keep = True
                saved_positive += 1
            else:
                if rng.random() < keep_empty_ratio:
                    keep = True
                    saved_empty += 1

            if not keep:
                continue

            fname = f"patch_{patch_id:05d}.png"
            Image.fromarray(img.astype(np.uint8)).save(out_img_dir / fname)
            Image.fromarray((mask > 0).astype(np.uint8) * 255).save(out_mask_dir / fname)

            patch_id += 1
            saved += 1

print("Saved total patches:", saved)
print("Saved positive patches:", saved_positive)
print("Saved mostly empty patches:", saved_empty)
print("Images dir:", out_img_dir)
print("Masks dir:", out_mask_dir)