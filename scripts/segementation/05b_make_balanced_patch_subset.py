import os
import glob
import shutil
import random
from pathlib import Path

import numpy as np
import rasterio

random.seed(42)
np.random.seed(42)

# ===== paths =====
IMG_DIR = Path("data/seg/images")
MASK_DIR = Path("data/seg/masks")

OUT_IMG_DIR = Path("data/patches_balanced/images")
OUT_MASK_DIR = Path("data/patches_balanced/masks")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MASK_DIR.mkdir(parents=True, exist_ok=True)

# ===== params =====
FOREGROUND_THRESHOLD = 0.01   # patch中前景占比超过1%视为positive
NEG_POS_RATIO = 1.5           # 负样本保留数量 = 正样本数量 * 1.5

def read_mask_fraction(mask_path):
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
    arr = (arr > 0).astype(np.uint8)
    return arr.mean()

def get_mask_path_from_image(img_path):
    return MASK_DIR / Path(img_path).name

def main():
    image_files = sorted(glob.glob(str(IMG_DIR / "*")))
    print(f"Found {len(image_files)} image patches")

    positives = []
    negatives = []

    for img_fp in image_files:
        mask_fp = get_mask_path_from_image(img_fp)
        if not mask_fp.exists():
            print(f"Missing mask for {img_fp}, skip.")
            continue

        fg_frac = read_mask_fraction(mask_fp)

        record = {
            "img": Path(img_fp),
            "mask": mask_fp,
            "fg_frac": fg_frac
        }

        if fg_frac >= FOREGROUND_THRESHOLD:
            positives.append(record)
        else:
            negatives.append(record)

    print(f"Positive patches: {len(positives)}")
    print(f"Negative patches: {len(negatives)}")

    n_neg_keep = min(len(negatives), int(len(positives) * NEG_POS_RATIO))
    kept_negatives = random.sample(negatives, n_neg_keep) if n_neg_keep > 0 else []

    selected = positives + kept_negatives
    random.shuffle(selected)

    print(f"Selected total: {len(selected)}")

    for rec in selected:
        shutil.copy2(rec["img"], OUT_IMG_DIR / rec["img"].name)
        shutil.copy2(rec["mask"], OUT_MASK_DIR / rec["mask"].name)

    print("Balanced subset saved to:")
    print(OUT_IMG_DIR)
    print(OUT_MASK_DIR)

if __name__ == "__main__":
    main()