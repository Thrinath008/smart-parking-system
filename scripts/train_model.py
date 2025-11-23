#!/usr/bin/env python3
"""
train_model.py

Train a parking-slot occupancy classifier (empty vs occupied)
using the images collected via collect_data.py.

Expected directory structure:

data/raw/
  A1/
    empty/
      *.jpg
    occupied/
      *.jpg
  A2/
    empty/
    occupied/
  ...

We do NOT care about the slot IDs themselves; every image is
just labeled based on its folder (empty / occupied).

Outputs:
  - models/trained/slot_classifier_best.pth
"""

import os
import argparse
from glob import glob
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from PIL import Image

from torchvision import transforms, models

from tqdm import tqdm


DATA_ROOT = "data/raw"
MODEL_DIR = "models/trained"


# ---------- Dataset ----------

class ParkingSlotDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 classes: Tuple[str, ...] = ("empty", "occupied"),
                 img_size: int = 224,
                 train: bool = True):
        self.root_dir = root_dir
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.img_size = img_size

        # Collect all image paths and labels
        self.samples: List[Tuple[str, int]] = []

        for slot_dir in sorted(os.listdir(root_dir)):
            slot_path = os.path.join(root_dir, slot_dir)
            if not os.path.isdir(slot_path):
                continue
            for cls in classes:
                cls_dir = os.path.join(slot_path, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for img_path in glob(os.path.join(cls_dir, ext)):
                        self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(f"No images found in {root_dir} for classes {classes}")

        # Basic transforms: augmentation for train, light for val
        if train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label


# ---------- Model ----------

def create_model(num_classes: int = 2):
    # Use a small pretrained model: MobileNetV3 Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# ---------- Training utilities ----------

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val  ", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ---------- Main train function ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=DATA_ROOT,
                        help="Root directory of raw data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = get_device()
    print(f"[INFO] Using device: {device}")

    print("[INFO] Building datasets...")
    full_dataset = ParkingSlotDataset(root_dir=args.data_root,
                                      classes=("empty", "occupied"),
                                      img_size=args.img_size,
                                      train=True)

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # For validation, we want deterministic, no augmentation transforms
    # So rebuild val_dataset with train=False but same indices
    val_indices = val_dataset.indices if hasattr(val_dataset, "indices") else range(len(val_dataset))
    val_samples = [full_dataset.samples[i] for i in val_indices]
    val_dataset = ParkingSlotDataset(root_dir=args.data_root,
                                     classes=("empty", "occupied"),
                                     img_size=args.img_size,
                                     train=False)
    # Override val_dataset.samples with selected ones
    val_dataset.samples = val_samples

    print(f"[INFO] Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    model = create_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    best_model_path = os.path.join(MODEL_DIR, "slot_classifier_best.pth")

    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        print(f"[INFO] Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        print(f"[INFO] Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": {"empty": 0, "occupied": 1},
                "img_size": args.img_size,
            }, best_model_path)
            print(f"[INFO] New best model saved -> {best_model_path} (val_acc={val_acc:.4f})")

    print(f"\n[INFO] Training done. Best val acc: {best_val_acc:.4f}")
    print(f"[INFO] Best model: {best_model_path}")


if __name__ == "__main__":
    main()