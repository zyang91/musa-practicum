import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

import segmentation_models_pytorch as smp
from tqdm import tqdm

# ===== config =====
TRAIN_IMG_DIR = Path("data/split/train/images")
TRAIN_MASK_DIR = Path("data/split/train/masks")
VAL_IMG_DIR = Path("data/split/val/images")
VAL_MASK_DIR = Path("data/split/val/masks")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
NUM_EPOCHS = 25
LR = 1e-4
IMG_CHANNELS = 3
ENCODER = "resnet34"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== dataset =====
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
            img = src.read().astype(np.float32)   # (C,H,W)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32) # (H,W)

        # normalize image to [0,1]
        if img.max() > 0:
            img = img / img.max()

        mask = (mask > 0).astype(np.float32)

        img = torch.tensor(img, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

        return img, mask

# ===== metrics =====
def dice_coef(preds, targets, threshold=0.5, eps=1e-7):
    preds = (preds > threshold).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()

# ===== training loops =====
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_probs = []
    all_masks = []

    for imgs, masks in tqdm(loader, desc="Val", leave=False):
        imgs = imgs.to(device)
        masks = masks.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, masks)

        probs = torch.sigmoid(outputs)

        running_loss += loss.item() * imgs.size(0)
        all_probs.append(probs.cpu())
        all_masks.append(masks.cpu())

    val_loss = running_loss / len(loader.dataset)
    all_probs = torch.cat(all_probs, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    val_dice = dice_coef(all_probs, all_masks, threshold=0.5)

    return val_loss, val_dice

def main():
    print("Device:", DEVICE)

    train_ds = RasterSegDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    val_ds = RasterSegDataset(VAL_IMG_DIR, VAL_MASK_DIR)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=IMG_CHANNELS,
        classes=1
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_dice = -1
    history = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_dice = eval_one_epoch(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Dice:   {val_dice:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice
        })

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_DIR / "best_unet.pt")
            print("Saved best model.")

    pd.DataFrame(history).to_csv(MODEL_DIR / "training_history.csv", index=False)
    print("Training finished.")
    print(f"Best val dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()