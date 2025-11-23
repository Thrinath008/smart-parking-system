#!/usr/bin/env python3
import json, os, sys

PATH = "data/processed/rois.json"
if not os.path.exists(PATH):
    print("No rois.json found at", PATH)
    sys.exit(0)

with open(PATH, "r") as f:
    rois = json.load(f)

good = []
bad = []
for r in rois:
    bbox = r.get("bbox", [0,0,0,0])
    w = bbox[2] if len(bbox) > 2 else 0
    h = bbox[3] if len(bbox) > 3 else 0
    if w > 10 and h > 10:   # threshold to avoid near-zero boxes
        good.append(r)
    else:
        bad.append(r)

with open(PATH, "w") as f:
    json.dump(good, f, indent=2)

print(f"Sanitized ROIs. Kept {len(good)} entries, removed {len(bad)} invalid entries.")
if bad:
    print("Removed IDs:", [r.get('id') for r in bad])