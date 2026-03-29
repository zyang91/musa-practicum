import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp


# =========================
# Config
# =========================
TEST_IMG_DIR = Path("data/split/test/images")
TEST_MASK_DIR = Path("data/split/test/masks")
MODEL_PATH = Path("./models/best_unet.pt")
OUTPUT_DIR = Path("outputs/visualizations")

THRESHOLD = 0.45
NUM_SAMPLES = 12
IMG_SIZE = 256   # if your training used another size, change here
ENCODER_NAME = "resnet34"   # make sure this matches 06_train_unet.py
ENCODER_WEIGHTS = None      # None because we are loading trained weights
BATCH_SIZE = 1
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_image(path):
    with rasterio.open(path) as src:
        img = src.read()  # (C, H, W)
    img = np.transpose(img, (1, 2, 0))  # (H, W, C)

    # assume RGB tif patches
    if img.dtype != np.float32:
        img = img.astype(np.float32)

    # simple normalization to 0-1
    if img.max() > 1.0:
        img = img / 255.0

    img = np.clip(img, 0.0, 1.0)
    return img


def read_mask(path):
    with rasterio.open(path) as src:
        mask = src.read(1)
    mask = (mask > 0).astype(np.float32)
    return mask


def tensor_from_image(img):
    # (H, W, C) -> (C, H, W)
    x = np.transpose(img, (2, 0, 1)).astype(np.float32)
    return torch.from_numpy(x)


def dice_score_np(y_true, y_pred, eps=1e-7):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    inter = (y_true * y_pred).sum()
    return (2 * inter + eps) / (y_true.sum() + y_pred.sum() + eps)


def iou_score_np(y_true, y_pred, eps=1e-7):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)
    inter = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - inter
    return (inter + eps) / (union + eps)


def make_overlay(img, pred_mask, alpha=0.35):
    overlay = img.copy()
    if overlay.ndim == 2:
        overlay = np.stack([overlay] * 3, axis=-1)

    red = np.zeros_like(overlay)
    red[..., 0] = 1.0

    pred_mask_3d = pred_mask[..., None]
    overlay = np.where(
        pred_mask_3d > 0,
        overlay * (1 - alpha) + red * alpha,
        overlay
    )
    return np.clip(overlay, 0, 1)


# =========================
# Dataset
# =========================
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)

        self.image_paths = sorted(list(self.img_dir.glob("*")))
        self.mask_paths = [self.mask_dir / p.name for p in self.image_paths]

        missing = [m for m in self.mask_paths if not m.exists()]
        if missing:
            raise FileNotFoundError(f"Missing masks, example: {missing[0]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = read_image(img_path)
        mask = read_mask(mask_path)

        x = tensor_from_image(img)
        y = torch.from_numpy(mask).unsqueeze(0).float()

        return x, y, str(img_path), img, mask


# =========================
# Model
# =========================
def build_model():
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,
    )
    return model


def load_model(model_path):
    model = build_model()

    ckpt = torch.load(model_path, map_location=DEVICE)

    # support either plain state_dict or wrapped checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return model


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = SegDataset(TEST_IMG_DIR, TEST_MASK_DIR)

    print(f"Using device: {DEVICE}")
    print(f"Test samples: {len(dataset)}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Threshold: {THRESHOLD}")

    if len(dataset) == 0:
        raise ValueError("No test samples found.")

    model = load_model(MODEL_PATH)

    sample_indices = list(range(len(dataset)))
    random.shuffle(sample_indices)
    sample_indices = sample_indices[:min(NUM_SAMPLES, len(dataset))]

    metrics = []

    for i, idx in enumerate(sample_indices, start=1):
        x, y, img_path, img_np, mask_np = dataset[idx]

        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(DEVICE))
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        pred_bin = (probs >= THRESHOLD).astype(np.uint8)

        dice = dice_score_np(mask_np, pred_bin)
        iou = iou_score_np(mask_np, pred_bin)
        metrics.append((Path(img_path).name, dice, iou))

        overlay = make_overlay(img_np, pred_bin)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(img_np)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask_np, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_bin, cmap="gray")
        axes[2].set_title(f"Prediction\nDice={dice:.3f}, IoU={iou:.3f}")
        axes[2].axis("off")

        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")
        axes[3].axis("off")

        plt.tight_layout()

        out_path = OUTPUT_DIR / f"sample_{i:02d}_{Path(img_path).stem}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

        print(f"Saved: {out_path}")

    print("\nSample metrics:")
    for name, dice, iou in metrics:
        print(f"{name}: Dice={dice:.4f}, IoU={iou:.4f}")

    mean_dice = np.mean([m[1] for m in metrics])
    mean_iou = np.mean([m[2] for m in metrics])
    print(f"\nMean Dice (sampled): {mean_dice:.4f}")
    print(f"Mean IoU  (sampled): {mean_iou:.4f}")
    print(f"\nDone. Visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()