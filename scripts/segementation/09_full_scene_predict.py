from pathlib import Path
import math
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import segmentation_models_pytorch as smp


# =========================
# Config
# =========================
INPUT_RASTER = Path("data/processed/university_city_pilot_mosaic.tif")
MODEL_PATH = Path("./models/best_unet.pt")
OUTPUT_DIR = Path("outputs/full_scene")

PATCH_SIZE = 256
STRIDE = 128              # overlap prediction; 128 is safe
THRESHOLD = 0.45

ENCODER_NAME = "resnet34"   # must match training
IN_CHANNELS = 3
CLASSES = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Model
# =========================
def build_model():
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
    )
    return model


def load_model(model_path):
    model = build_model()
    ckpt = torch.load(model_path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return model


# =========================
# Helpers
# =========================
def normalize_img(img):
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)
    return img


def pad_patch(arr, patch_size):
    """
    arr: (C, H, W)
    returns padded arr and original h,w
    """
    c, h, w = arr.shape
    out = np.zeros((c, patch_size, patch_size), dtype=arr.dtype)
    out[:, :h, :w] = arr
    return out, h, w


def get_positions(full_size, patch_size, stride):
    if full_size <= patch_size:
        return [0]

    positions = list(range(0, full_size - patch_size + 1, stride))
    if positions[-1] != full_size - patch_size:
        positions.append(full_size - patch_size)
    return positions


# =========================
# Main
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prob_out = OUTPUT_DIR / f"{INPUT_RASTER.stem}_prob.tif"
    bin_out = OUTPUT_DIR / f"{INPUT_RASTER.stem}_bin.tif"

    print("Using device:", DEVICE)
    print("Input raster:", INPUT_RASTER)
    print("Model path:", MODEL_PATH)
    print("Output dir:", OUTPUT_DIR)
    print("Patch size:", PATCH_SIZE)
    print("Stride:", STRIDE)
    print("Threshold:", THRESHOLD)

    model = load_model(MODEL_PATH)

    with rasterio.open(INPUT_RASTER) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        count = src.count
        transform = src.transform
        crs = src.crs

        if count < 3:
            raise ValueError(f"Expected at least 3 bands, got {count}")

        ys = get_positions(height, PATCH_SIZE, STRIDE)
        xs = get_positions(width, PATCH_SIZE, STRIDE)

        print(f"Raster size: {width} x {height}")
        print(f"Total windows: {len(xs) * len(ys)}")

        prob_sum = np.zeros((height, width), dtype=np.float32)
        prob_count = np.zeros((height, width), dtype=np.float32)

        n_total = len(xs) * len(ys)
        n_done = 0

        for y in ys:
            for x in xs:
                window = Window(x, y, min(PATCH_SIZE, width - x), min(PATCH_SIZE, height - y))
                patch = src.read([1, 2, 3], window=window)   # (C,H,W)

                patch, h0, w0 = pad_patch(patch, PATCH_SIZE)
                patch = normalize_img(patch)

                tensor = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()   # (PATCH_SIZE, PATCH_SIZE)

                probs = probs[:h0, :w0]

                prob_sum[y:y+h0, x:x+w0] += probs
                prob_count[y:y+h0, x:x+w0] += 1.0

                n_done += 1
                if n_done % 100 == 0 or n_done == n_total:
                    print(f"Processed {n_done}/{n_total} windows")

    prob_avg = prob_sum / np.maximum(prob_count, 1e-6)
    pred_bin = (prob_avg >= THRESHOLD).astype(np.uint8)

    prob_profile = profile.copy()
    prob_profile.update(
        dtype="float32",
        count=1,
        compress="lzw"
    )

    bin_profile = profile.copy()
    bin_profile.update(
        dtype="uint8",
        count=1,
        compress="lzw",
        nodata=0
    )

    with rasterio.open(prob_out, "w", **prob_profile) as dst:
        dst.write(prob_avg.astype(np.float32), 1)

    with rasterio.open(bin_out, "w", **bin_profile) as dst:
        dst.write(pred_bin.astype(np.uint8), 1)

    print("\nSaved probability raster:", prob_out)
    print("Saved binary raster:", bin_out)
    print("Done.")


if __name__ == "__main__":
    main()