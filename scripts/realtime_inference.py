#!/usr/bin/env python3
"""
realtime_inference.py

Use the trained parking-slot classifier to run real-time inference
on the camera feed. For each ROI (A1..B4) defined in rois.json, we:

- Crop the region from the frame
- Run it through the trained model
- Predict: empty / occupied
- Draw a green box for empty, red for occupied, with label text

Usage:
  python scripts/realtime_inference.py --camera 0
"""

import os
import json
import argparse
from typing import Dict

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

import numpy as np


ROIS_PATH = "data/processed/rois.json"
MODEL_PATH = "models/trained/slot_classifier_best.pth"


def ensure_file(path: str, msg: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{msg}: {path}")


def load_rois(path: str):
    ensure_file(path, "ROIs file not found")
    with open(path, "r") as f:
        rois = json.load(f)
    if not rois:
        raise RuntimeError("ROIs list is empty. Run collect_data.py in define mode first.")
    return rois


def create_model(num_classes: int = 2):
    model = models.mobilenet_v3_small(weights=None)  # we'll load our own weights
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_checkpoint(model_path: str, device) -> Dict:
    ensure_file(model_path, "Model checkpoint not found")
    ckpt = torch.load(model_path, map_location=device)
    return ckpt


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--score-thresh", type=float, default=0.5,
                        help="Optional confidence threshold (currently just for display)")
    args = parser.parse_args()

    device = get_device()
    print(f"[INFO] Using device: {device}")

    # Load ROIs
    rois = load_rois(ROIS_PATH)
    print(f"[INFO] Loaded {len(rois)} ROIs from {ROIS_PATH}")

    # Load checkpoint
    ckpt = load_checkpoint(MODEL_PATH, device)
    class_to_idx = ckpt.get("class_to_idx", {"empty": 0, "occupied": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    img_size = ckpt.get("img_size", 224)

    # Build model
    model = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform(img_size)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        return

    print("[INFO] Real-time inference started.")
    print("Controls: 'q' or ESC to quit, 'p' to pause/resume.")

    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera.")
                break

            display = frame.copy()
            h0, w0 = frame.shape[:2]

            slot_status = []  # list of (id, pred_class, prob)

            for r in rois:
                sid = r["id"]
                x, y, w, h = r["bbox"]

                # clip bbox within frame
                x2 = max(0, min(x + w, w0))
                y2 = max(0, min(y + h, h0))
                x1 = max(0, min(x, w0 - 1))
                y1 = max(0, min(y, h0 - 1))

                if x1 >= x2 or y1 >= y2:
                    # invalid / out-of-frame ROI
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(display, f"{sid}:INVALID", (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    continue

                crop = frame[y1:y2, x1:x2]
                # BGR -> RGB
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

                # PIL-style transform expects HWC uint8
                pil_like = Image_from_numpy(crop_rgb)
                input_tensor = transform(pil_like).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)
                    pred_class = idx_to_class[int(pred_idx)]
                    conf_val = float(conf.item())

                slot_status.append((sid, pred_class, conf_val))

                # Choose color: green for empty, red for occupied
                if pred_class == "empty":
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                label_text = f"{sid}:{pred_class} {conf_val:.2f}"
                cv2.putText(display, label_text, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Optional overlay with summary
            empty_count = sum(1 for (_, c, _) in slot_status if c == "empty")
            occ_count = sum(1 for (_, c, _) in slot_status if c == "occupied")
            summary = f"Empty: {empty_count} | Occupied: {occ_count}"
            cv2.rectangle(display, (10, 10), (10 + 260, 40), (0, 0, 0), -1)
            cv2.putText(display, summary, (18, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Parking Occupancy - Live", display)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), 27):
            break
        elif k == ord('p'):
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference session ended.")


# Helper: minimal drop-in to wrap numpy array as PIL.Image-like object
from PIL import Image

def Image_from_numpy(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


if __name__ == "__main__":
    main()