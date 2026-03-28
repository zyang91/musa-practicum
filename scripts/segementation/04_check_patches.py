from pathlib import Path

img_dir = Path("data/seg/images")
mask_dir = Path("data/seg/masks")

imgs = sorted(img_dir.glob("*.png"))
masks = sorted(mask_dir.glob("*.png"))

print("Image patches:", len(imgs))
print("Mask patches:", len(masks))

if len(imgs) > 0:
    print("First 5 image patches:")
    for p in imgs[:5]:
        print(" ", p.name)

if len(masks) > 0:
    print("First 5 mask patches:")
    for p in masks[:5]:
        print(" ", p.name)