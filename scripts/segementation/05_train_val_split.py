from pathlib import Path
import random
import shutil

random.seed(42)

img_dir = Path("data/seg/images")
mask_dir = Path("data/seg/masks")

train_img_dir = Path("data/seg_split/images_train")
train_mask_dir = Path("data/seg_split/masks_train")
val_img_dir = Path("data/seg_split/images_val")
val_mask_dir = Path("data/seg_split/masks_val")

for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir]:
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):
        f.unlink()

imgs = sorted(img_dir.glob("*.png"))
assert len(imgs) > 0, "No image patches found."

names = [p.name for p in imgs]
random.shuffle(names)

val_ratio = 0.2
n_val = int(len(names) * val_ratio)

val_names = set(names[:n_val])
train_names = set(names[n_val:])

for name in train_names:
    shutil.copy2(img_dir / name, train_img_dir / name)
    shutil.copy2(mask_dir / name, train_mask_dir / name)

for name in val_names:
    shutil.copy2(img_dir / name, val_img_dir / name)
    shutil.copy2(mask_dir / name, val_mask_dir / name)

print("Train patches:", len(train_names))
print("Val patches:", len(val_names))