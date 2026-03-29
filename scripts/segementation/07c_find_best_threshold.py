from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import Dataset, DataLoader
import torch
import segmentation_models_pytorch as smp

# ===== config =====
VAL_IMG_DIR = Path("data/split/val/images")
VAL_MASK_DIR = Path("data/split/val/masks")
MODEL_PATH = Path("models/best_unet.pt")
OUT_CSV = Path("models/threshold_search.csv")

BATCH_SIZE = 8
IMG_CHANNELS = 3
ENCODER = "resnet34"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class RasterSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.files = sorted([p.name for p in self.img_dir.glob("*")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = self.img_dir / fname
        mask_path = self.mask_dir / fname

        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        if img.max() > 0:
            img = img / img.max()

        mask = (mask > 0).astype(np.float32)

        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        )

def calc_stats(prob, target, threshold, eps=1e-7):
    pred = (prob > threshold).astype(np.uint8)
    target = target.astype(np.uint8)

    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice
    }

@torch.no_grad()
def main():
    ds = RasterSegDataset(VAL_IMG_DIR, VAL_MASK_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        in_channels=IMG_CHANNELS,
        classes=1
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    probs_all = []
    masks_all = []

    for imgs, masks in loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs).cpu().numpy()
        masks = masks.cpu().numpy()

        probs_all.append(probs)
        masks_all.append(masks)

    probs_all = np.concatenate(probs_all, axis=0)
    masks_all = np.concatenate(masks_all, axis=0)

    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []

    for th in thresholds:
        rows.append(calc_stats(probs_all, masks_all, th))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    best_row = df.loc[df["f1"].idxmax()]
    print("Best threshold by F1:")
    print(best_row)

if __name__ == "__main__":
    main()