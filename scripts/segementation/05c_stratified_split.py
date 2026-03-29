import os
import glob
import shutil
from pathlib import Path

import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

# ===== paths =====
IMG_DIR = Path("data/patches_balanced/images")
MASK_DIR = Path("data/patches_balanced/masks")

OUT_ROOT = Path("data/split")
TRAIN_IMG = OUT_ROOT / "train/images"
TRAIN_MASK = OUT_ROOT / "train/masks"
VAL_IMG = OUT_ROOT / "val/images"
VAL_MASK = OUT_ROOT / "val/masks"
TEST_IMG = OUT_ROOT / "test/images"
TEST_MASK = OUT_ROOT / "test/masks"

for p in [TRAIN_IMG, TRAIN_MASK, VAL_IMG, VAL_MASK, TEST_IMG, TEST_MASK]:
    p.mkdir(parents=True, exist_ok=True)

# ===== params =====
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

def read_mask_fraction(mask_path):
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
    arr = (arr > 0).astype(np.uint8)
    return arr.mean()

def make_strata(mask_fraction):
    # 简单分层：纯背景 / 低前景 / 中前景 / 高前景
    if mask_fraction == 0:
        return 0
    elif mask_fraction < 0.01:
        return 1
    elif mask_fraction < 0.05:
        return 2
    else:
        return 3

def copy_pairs(file_list, split_name):
    img_out = OUT_ROOT / split_name / "images"
    mask_out = OUT_ROOT / split_name / "masks"

    for img_fp in file_list:
        img_fp = Path(img_fp)
        mask_fp = MASK_DIR / img_fp.name
        shutil.copy2(img_fp, img_out / img_fp.name)
        shutil.copy2(mask_fp, mask_out / mask_fp.name)

def main():
    image_files = sorted(glob.glob(str(IMG_DIR / "*")))
    print(f"Found {len(image_files)} balanced image patches")

    strata = []
    usable_files = []

    for img_fp in image_files:
        mask_fp = MASK_DIR / Path(img_fp).name
        if not mask_fp.exists():
            print(f"Missing mask for {img_fp}, skip.")
            continue

        frac = read_mask_fraction(mask_fp)
        strata.append(make_strata(frac))
        usable_files.append(img_fp)

    usable_files = np.array(usable_files)
    strata = np.array(strata)

    # 先切出 test
    train_val_files, test_files, train_val_strata, _ = train_test_split(
        usable_files,
        strata,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata
    )

    # 再从 train_val 切 val
    val_ratio_adjusted = VAL_SIZE / (1 - TEST_SIZE)

    train_files, val_files, _, _ = train_test_split(
        train_val_files,
        train_val_strata,
        test_size=val_ratio_adjusted,
        random_state=RANDOM_STATE,
        stratify=train_val_strata
    )

    print(f"Train: {len(train_files)}")
    print(f"Val: {len(val_files)}")
    print(f"Test: {len(test_files)}")

    copy_pairs(train_files, "train")
    copy_pairs(val_files, "val")
    copy_pairs(test_files, "test")

    print("Done. Split saved to data/split/")

if __name__ == "__main__":
    main()