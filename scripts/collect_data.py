#!/usr/bin/env python3
"""
collect_data.py

Two-step utility for the parking-slot dataset:

1) Define ROIs:
   python scripts/collect_data.py --mode define --camera 0
   - Draw rectangles with mouse (drag) and press Enter to name + save each ROI.
   - Press 'q' or ESC to finish. ROIs saved to data/processed/rois.json

2) Capture labeled images:
   python scripts/collect_data.py --mode capture --camera 0 --label empty
   - Uses saved ROIs; shows live feed with ROI overlays and spot previews.
   - Press SPACE to save current frame crops (one image per ROI) to data/raw/<spot_id>/<label>/
   - Press 'l' to list counts, 's' to change label, 'q' or ESC to quit.

Dependencies:
  - opencv-python
  - numpy
"""

import cv2
import json
import argparse
import os
from datetime import datetime

ROIS_PATH = "data/processed/rois.json"
RAW_DATA_DIR = "data/raw"

# ---------- Helper functions ----------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_rois(rois):
    ensure_dir(os.path.dirname(ROIS_PATH) or ".")
    with open(ROIS_PATH, "w") as f:
        json.dump(rois, f, indent=2)
    print(f"[INFO] Saved {len(rois)} ROIs -> {ROIS_PATH}")


def load_rois():
    if not os.path.exists(ROIS_PATH):
        return []
    with open(ROIS_PATH, "r") as f:
        return json.load(f)


def next_filename(spot_dir):
    ensure_dir(spot_dir)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S%f")[:-3]
    return os.path.join(spot_dir, f"{ts}.jpg")


# ---------- ROI drawing tool ----------
class ROIEditor:
    def __init__(self, winname="define_rois"):
        self.win = winname
        self.drawing = False
        self.ix = self.iy = -1
        self.rect = None
        self.rois = []  # list of dicts: {"id": "S01", "bbox": [x,y,w,h] }
        self.frame = None

        cv2.namedWindow(self.win)
        cv2.setMouseCallback(self.win, self.mouse_cb)

    def mouse_cb(self, event, x, y, flags, param):
        # Mouse callback to draw rectangles
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.rect = None
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                x0, y0 = self.ix, self.iy
                self.rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x0, y0 = self.ix, self.iy
            self.rect = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            x, y, w, h = map(int, self.rect)
            # require a minimum size to avoid zero-area ROIs
            if w < 20 or h < 20:
                print("[WARN] Rectangle too small — please drag a larger region.")
                self.rect = None
                return
            # ask for an id via terminal
            try:
                spot_id = input("Enter spot id (e.g. S01) for this ROI (or press Enter to ignore): ").strip()
            except EOFError:
                spot_id = ""
            if spot_id:
                # avoid duplicate ids
                existing_ids = [r['id'] for r in self.rois]
                if spot_id in existing_ids:
                    print(f"[WARN] ID {spot_id} already used — choose a unique id.")
                else:
                    self.rois.append({"id": spot_id, "bbox": [x, y, w, h]})
                    print(f"[INFO] Added ROI {spot_id}: {(x, y, w, h)}")
            else:
                print("[INFO] Ignored ROI (no id provided).")
            self.rect = None

    def run(self, cap_index=0):
        cap = cv2.VideoCapture(cap_index)
        if not cap.isOpened():
            print("[ERROR] Unable to open camera. Try a different --camera index.")
            return
        print("[INFO] ROI define mode: draw rectangles with mouse. After drawing release, enter spot id in terminal.")
        print("Press 'q' or ESC to finish and save ROIs. Press Ctrl+C to cancel.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                self.frame = frame.copy()
                display = frame.copy()
                # draw existing ROIs
                for r in self.rois:
                    x, y, w, h = r['bbox']
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display, r['id'], (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                # draw current rect
                if self.rect is not None:
                    x, y, w, h = map(int, self.rect)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 200, 255), 2)
                cv2.imshow(self.win, display)
                k = cv2.waitKey(1) & 0xFF
                if k in [ord('q'), 27]:  # q or ESC
                    break
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            # Save ROIs only if we have at least one valid ROI
            if self.rois:
                save_rois(self.rois)
            else:
                print("[WARN] No ROIs defined; nothing saved.")


# ---------- Capture mode ----------
def capture_mode(camera_index=0, label="empty"):
    rois = load_rois()
    if len(rois) == 0:
        print(f"[ERROR] No ROIs found at {ROIS_PATH}. Run define mode first.")
        return
    # Validate ROIs are within frame bounds later
    # Create label directories
    for r in rois:
        spot_dir = os.path.join(RAW_DATA_DIR, r['id'], label)
        ensure_dir(spot_dir)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERROR] Unable to open camera.")
        return

    print("[INFO] Capture mode. Press SPACE to save crops for all ROIs with current label.")
    print("Press 'l' to list counts, 's' to change label, 'q' or ESC to quit.")

    current_label = label
    counters = {r['id']: len(os.listdir(os.path.join(RAW_DATA_DIR, r['id'], current_label))) if os.path.exists(os.path.join(RAW_DATA_DIR, r['id'], current_label)) else 0 for r in rois}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        previews = []
        # draw ROIs and small preview thumbnails
        for idx, r in enumerate(rois):
            x, y, w, h = r['bbox']
            # ensure bbox within frame bounds
            h0, w0 = frame.shape[:2]
            x2 = max(0, min(x + w, w0))
            y2 = max(0, min(y + h, h0))
            x1 = max(0, min(x, w0 - 1))
            y1 = max(0, min(y, h0 - 1))
            if x1 >= x2 or y1 >= y2:
                crop = None
            else:
                crop = frame[y1:y2, x1:x2].copy()
            # Draw rectangle and ID
            color = (0,255,0) if crop is not None else (0,0,255)
            cv2.rectangle(display, (x1,y1), (x2, y2), color, 2)
            label_text = f"{r['id']}:{current_label[:3]}"
            cv2.putText(display, label_text, (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # small preview
            if crop is not None:
                pv = cv2.resize(crop, (160, 90))
            else:
                pv = 255 * np.ones((90,160,3), dtype=np.uint8)
                cv2.putText(pv, 'INVALID', (8,45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            previews.append((r['id'], pv))
        # show previews on right side
        h0, w0 = display.shape[:2]
        for i, (pid, pv) in enumerate(previews):
            x0 = max(10, w0 - pv.shape[1] - 10)
            y0 = 10 + i*(pv.shape[0]+10)
            # ensure area fits
            if y0 + pv.shape[0] <= h0:
                display[y0:y0+pv.shape[0], x0:x0+pv.shape[1]] = pv
                cv2.putText(display, pid, (x0, y0+pv.shape[0]+14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("capture", display)
        k = cv2.waitKey(1) & 0xFF

        if k == 32:  # SPACE = save current crops
            saved = 0
            for r in rois:
                x, y, w, h = r['bbox']
                h0, w0 = frame.shape[:2]
                x2 = max(0, min(x + w, w0))
                y2 = max(0, min(y + h, h0))
                x1 = max(0, min(x, w0 - 1))
                y1 = max(0, min(y, h0 - 1))
                if x1 >= x2 or y1 >= y2:
                    print(f"[WARN] Skipping save for {r['id']} — ROI out of frame bounds.")
                    continue
                crop = frame[y1:y2, x1:x2].copy()
                spot_dir = os.path.join(RAW_DATA_DIR, r['id'], current_label)
                ensure_dir(spot_dir)
                filename = next_filename(spot_dir)
                cv2.imwrite(filename, crop)
                saved += 1
                counters[r['id']] = counters.get(r['id'], 0) + 1
            print(f"[INFO] Saved {saved} images for label '{current_label}' (one per valid ROI).")
        elif k == ord('l'):
            print("--- counts ---")
            for s, c in counters.items():
                print(f"{s}: {c}")
            print("--------------")
        elif k == ord('s'):
            newlabel = input("Enter new label name (e.g. empty / occupied / occluded): ").strip()
            if newlabel:
                current_label = newlabel
                # ensure dirs exist
                for r in rois:
                    ensure_dir(os.path.join(RAW_DATA_DIR, r['id'], current_label))
                print(f"[INFO] Switched label -> {current_label}")
        elif k in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Capture session ended. Totals:")
    for s, c in counters.items():
        print(f" - {s}: {c}")


# ---------- main CLI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["define", "capture"], required=True, help="Mode: define ROIs or capture images")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (0 is built-in / default)")
    parser.add_argument("--label", type=str, default="empty", help="Label to save images under (capture mode)")
    args = parser.parse_args()

    ensure_dir("data/processed")
    ensure_dir(RAW_DATA_DIR)

    if args.mode == "define":
        editor = ROIEditor()
        editor.run(cap_index=args.camera)
    elif args.mode == "capture":
        capture_mode(camera_index=args.camera, label=args.label)
    else:
        print("[ERROR] unknown mode")


if __name__ == "__main__":
    main()