"""
03_predict.py
Run UNet sliding-window prediction on a neighborhood mosaic.

Reads   : data/processed/<neighborhood>_mosaic.tif
Model   : models/best_unet.pt  (ResNet34 encoder, trained on University City)
Outputs : result/<neighborhood>/outputs/<neighborhood>_mosaic_prob.tif
          result/<neighborhood>/outputs/<neighborhood>_mosaic_bin.tif
"""

from pathlib import Path
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import segmentation_models_pytorch as smp

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path("models/best_unet.pt")
PROCESSED_DIR = Path("data/processed")
RESULT_ROOT = Path("result")

PATCH_SIZE = 256
STRIDE = 128
THRESHOLD = 0.45

ENCODER_NAME = "resnet34"
IN_CHANNELS = 3
CLASSES = 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Model ───────────────────────────────────────────────────────────────────
def load_model(model_path: Path):
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,
        in_channels=IN_CHANNELS,
        classes=CLASSES,
    )
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model


# ── Helpers ─────────────────────────────────────────────────────────────────
def normalize_img(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return np.clip(img, 0.0, 1.0)


def pad_patch(arr: np.ndarray, patch_size: int):
    """Pad (C, H, W) array to (C, patch_size, patch_size). Returns padded, orig_h, orig_w."""
    c, h, w = arr.shape
    out = np.zeros((c, patch_size, patch_size), dtype=arr.dtype)
    out[:, :h, :w] = arr
    return out, h, w


def tile_positions(full_size: int, patch_size: int, stride: int):
    if full_size <= patch_size:
        return [0]
    positions = list(range(0, full_size - patch_size + 1, stride))
    if positions[-1] != full_size - patch_size:
        positions.append(full_size - patch_size)
    return positions


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run UNet prediction on neighborhood mosaic")
    parser.add_argument("--name", required=True, help="Neighborhood name")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Binary threshold (default: {THRESHOLD})")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Inference batch size (default: 16)")
    args = parser.parse_args()

    name = args.name
    threshold = args.threshold
    batch_size = args.batch_size

    input_raster = PROCESSED_DIR / f"{name}_mosaic.tif"
    out_dir = RESULT_ROOT / name / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    prob_out = out_dir / f"{name}_mosaic_prob.tif"
    bin_out = out_dir / f"{name}_mosaic_bin.tif"

    print(f"Device       : {DEVICE}")
    print(f"Neighborhood : {name}")
    print(f"Input        : {input_raster}")
    print(f"Threshold    : {threshold}")
    print(f"Batch size   : {batch_size}")

    if not input_raster.exists():
        raise FileNotFoundError(f"Mosaic not found: {input_raster}")

    model = load_model(MODEL_PATH)

    with rasterio.open(input_raster) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        count = src.count

        if count < 3:
            raise ValueError(f"Expected ≥3 bands, got {count}")

        ys = tile_positions(height, PATCH_SIZE, STRIDE)
        xs = tile_positions(width, PATCH_SIZE, STRIDE)
        n_total = len(xs) * len(ys)
        print(f"Raster size  : {width} x {height}")
        print(f"Total windows: {n_total}")

        prob_sum = np.zeros((height, width), dtype=np.float32)
        prob_count = np.zeros((height, width), dtype=np.float32)

        # Collect all window specs for batched inference
        windows = []
        for y in ys:
            for x in xs:
                win_h = min(PATCH_SIZE, height - y)
                win_w = min(PATCH_SIZE, width - x)
                windows.append((x, y, win_w, win_h))

        n_done = 0
        for batch_start in range(0, len(windows), batch_size):
            batch_specs = windows[batch_start:batch_start + batch_size]
            patches = []
            orig_sizes = []

            for x, y, win_w, win_h in batch_specs:
                window = Window(x, y, win_w, win_h)
                patch = src.read([1, 2, 3], window=window)
                patch, h0, w0 = pad_patch(patch, PATCH_SIZE)
                patch = normalize_img(patch)
                patches.append(patch)
                orig_sizes.append((h0, w0))

            tensor = torch.from_numpy(np.stack(patches)).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

            for i, (x, y, _, _) in enumerate(batch_specs):
                h0, w0 = orig_sizes[i]
                prob_sum[y:y+h0, x:x+w0] += probs[i, :h0, :w0]
                prob_count[y:y+h0, x:x+w0] += 1.0

            n_done += len(batch_specs)
            if n_done % 200 == 0 or n_done == n_total:
                print(f"  Processed {n_done}/{n_total} windows")

    prob_avg = prob_sum / np.maximum(prob_count, 1e-6)
    pred_bin = (prob_avg >= threshold).astype(np.uint8)

    # Save probability raster
    prob_profile = profile.copy()
    prob_profile.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(prob_out, "w", **prob_profile) as dst:
        dst.write(prob_avg.astype(np.float32), 1)

    # Save binary raster
    bin_profile = profile.copy()
    bin_profile.update(dtype="uint8", count=1, compress="lzw", nodata=0)
    with rasterio.open(bin_out, "w", **bin_profile) as dst:
        dst.write(pred_bin, 1)

    pos_px = int(pred_bin.sum())
    print(f"\nPositive pixels: {pos_px}")
    print(f"Saved prob   : {prob_out}")
    print(f"Saved binary : {bin_out}")
    print("Done.")


if __name__ == "__main__":
    main()
