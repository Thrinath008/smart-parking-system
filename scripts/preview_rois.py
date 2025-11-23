#!/usr/bin/env python3
# scripts/preview_rois.py
import cv2, json, argparse, os

ROIS_PATH = "data/processed/rois.json"

def load_rois():
    if not os.path.exists(ROIS_PATH):
        raise SystemExit(f"No {ROIS_PATH} found. Run define mode first.")
    with open(ROIS_PATH, "r") as f:
        return json.load(f)

def draw_rois(frame, rois):
    out = frame.copy()
    for r in rois:
        x,y,w,h = r['bbox']
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(out, r['id'], (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
    return out

def preview_image(path):
    rois = load_rois()
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Could not open image: {path}")
    out = draw_rois(img, rois)
    cv2.imshow("rois_preview", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preview_camera(cam_index):
    rois = load_rois()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise SystemExit("Unable to open camera.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = draw_rois(frame, rois)
        cv2.imshow("rois_preview", out)
        k = cv2.waitKey(1) & 0xff
        if k in (27, ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to an image to preview")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for live preview")
    args = parser.parse_args()
    if args.image:
        preview_image(args.image)
    else:
        preview_camera(args.camera)
        