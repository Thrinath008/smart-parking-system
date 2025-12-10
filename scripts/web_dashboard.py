
#!/usr/bin/env python3
"""
web_dashboard.py

Enhanced dashboard with purpose-driven occupancy prediction + time predictor training endpoint.

Now includes:
- /api/train_predictor to train a small RandomForest regressor from synthetic + real logs
- Uses trained predictor (if available) when setting purpose to compute expected_end

Usage: python scripts/web_dashboard.py --camera 0
"""

import os
import json
import argparse
import threading
import time
import csv
from typing import Dict, List, Any
from datetime import datetime

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
from PIL import Image

from flask import Flask, jsonify, render_template_string, Response, request

# ML training imports (sklearn + pandas + joblib)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


ROIS_PATH = "data/processed/rois.json"
MODEL_PATH = "models/trained/slot_classifier_best.pth"
LOG_DIR = "data/logs"
LOG_CSV = os.path.join(LOG_DIR, "occupancy_log.csv")
SYNTH_CSV = os.path.join(LOG_DIR, "predictor_synthetic.csv")
PREDICTOR_PATH = "models/trained/time_predictor.pkl"

# Shared state between inference loop and Flask
slot_status: Dict[str, Dict] = {}  # per-slot info
latest_frame_jpeg = None  # type: ignore  # most recent annotated frame as JPEG bytes

# Internal tracking: start time per slot when it became occupied
slot_timers: Dict[str, float] = {}
# Simple in-memory log buffer (also persisted to CSV)
occupancy_logs: List[Dict[str, Any]] = []

# Purpose -> default duration (minutes)
PURPOSE_DEFAULT_MINUTES = {
    "shopping": 90,
    "eating": 60,
    "cinema": 180,
    "work": 240,
    "other": 30,
}


# ---------- Utilities ----------

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
    model = models.mobilenet_v3_small(weights=None)  # we load our own weights
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def get_device():
    # Keep CPU to keep inference stable on macOS for this demo
    return torch.device("cpu")


def build_transform(img_size: int):
    # Use a smaller input size for faster inference
    img_size = min(img_size, 160)
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def pil_from_numpy(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def ensure_log_dir():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_CSV):
        # create CSV with header
        with open(LOG_CSV, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "slot_id", "duration_seconds", "start_time", "end_time"]) 


def append_log_csv(slot_id: str, start_ts: float, end_ts: float, duration: float):
    ensure_log_dir()
    row = [datetime.utcfromtimestamp(end_ts).isoformat() + "Z", slot_id, f"{duration:.3f}",
           datetime.utcfromtimestamp(start_ts).isoformat() + "Z", datetime.utcfromtimestamp(end_ts).isoformat() + "Z"]
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


# ---------- Predictor training & inference ----------

def load_predictor():
    if os.path.exists(PREDICTOR_PATH):
        try:
            return joblib.load(PREDICTOR_PATH)
        except Exception:
            return None
    return None


def train_time_predictor(save_path: str = PREDICTOR_PATH, synth_path: str = SYNTH_CSV, real_path: str = LOG_CSV):
    """Train a small RandomForestRegressor using synthetic + real logs and save the pipeline."""
    # Read synthetic dataset if present
    frames = []
    if os.path.exists(synth_path):
        try:
            df_s = pd.read_csv(synth_path)
            df_s = df_s.rename(columns={
                'start_hour': 'start_hour', 'day_of_week': 'day_of_week',
                'purpose': 'purpose', 'duration_seconds': 'duration_seconds', 'slot_id': 'slot_id'
            })
            frames.append(df_s[['slot_id', 'purpose', 'start_hour', 'day_of_week', 'duration_seconds']])
        except Exception as e:
            print('[WARN] failed to read synthetic CSV:', e)

    # Read real occupancy logs
    if os.path.exists(real_path):
        try:
            df_r = pd.read_csv(real_path)
            # parse start_time to extract hour and weekday
            def parse_hour_day(s):
                try:
                    dt = datetime.fromisoformat(s.replace('Z', ''))
                    return dt.hour, dt.weekday()
                except Exception:
                    return 12, 0
            parsed = df_r['start_time'].apply(lambda s: parse_hour_day(s))
            df_r['start_hour'] = parsed.apply(lambda t: t[0])
            df_r['day_of_week'] = parsed.apply(lambda t: t[1])
            df_r = df_r.rename(columns={'duration_seconds': 'duration_seconds', 'slot_id': 'slot_id'})
            frames.append(df_r[['slot_id', 'purpose', 'start_hour', 'day_of_week', 'duration_seconds']])
        except Exception as e:
            print('[WARN] failed to read real logs:', e)

    if not frames:
        raise RuntimeError('No data available to train predictor. Generate synthetic data first.')

    df = pd.concat(frames, ignore_index=True)
    # Minimal cleaning
    df['purpose'] = df['purpose'].fillna('other')
    df['start_hour'] = df['start_hour'].astype(int)
    df['day_of_week'] = df['day_of_week'].astype(int)
    df['duration_seconds'] = df['duration_seconds'].astype(float)

    # Features: purpose (one-hot), start_hour, day_of_week
    X = df[['purpose', 'start_hour', 'day_of_week']]
    y = df['duration_seconds']

    # Build pipeline
    cat_cols = ['purpose']
    num_cols = ['start_hour', 'day_of_week']
    pre = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ], remainder='passthrough')

    model = Pipeline([
        ('pre', pre),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    model.fit(X_train, y_train)
    # Basic validation
    val_score = model.score(X_val, y_val)
    print(f"[INF] Trained predictor. Validation R^2: {val_score:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return save_path, val_score


def predict_duration_with_model(purpose: str, start_hour: int, day_of_week: int):
    mdl = load_predictor()
    if mdl is None:
        return None
    X = pd.DataFrame([{'purpose': purpose, 'start_hour': start_hour, 'day_of_week': day_of_week}])
    try:
        pred = mdl.predict(X)[0]
        return float(max(1.0, pred))
    except Exception:
        return None


# ---------- Inference loop (background thread) ----------

def inference_loop(camera_index: int = 0, poll_delay: float = 0.01):
    global slot_status, latest_frame_jpeg, slot_timers, occupancy_logs

    device = get_device()
    print(f"[INF] Inference using device: {device}")

    rois = load_rois(ROIS_PATH)
    print(f"[INF] Loaded {len(rois)} ROIs from {ROIS_PATH}")

    ensure_file(MODEL_PATH, "Model checkpoint not found")
    ckpt = torch.load(MODEL_PATH, map_location=device)
    class_to_idx = ckpt.get("class_to_idx", {"empty": 0, "occupied": 1})
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    img_size = ckpt.get("img_size", 224)

    model = create_model(num_classes=len(class_to_idx))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    transform = build_transform(img_size)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("[ERR] Unable to open camera for inference loop.")
        return

    # Use default camera settings for a stable, smooth preview (do not force FPS/resize)
    print("[INF] Inference loop started.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(1.0)
                continue

            display = frame.copy()
            h0, w0 = frame.shape[:2]

            # First pass: collect ROI crops/tensors to build a batch
            roi_entries = []  # list of dicts with sid, coords, tensor (or None if invalid)
            for r in rois:
                sid = r["id"]
                x, y, w, h = r["bbox"]

                # clip bbox within frame
                x2 = max(0, min(x + w, w0))
                y2 = max(0, min(y + h, h0))
                x1 = max(0, min(x, w0 - 1))
                y1 = max(0, min(y, h0 - 1))

                entry = {
                    "sid": sid,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "tensor": None,
                }

                if x1 >= x2 or y1 >= y2:
                    # invalid / out of frame
                    roi_entries.append(entry)
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = pil_from_numpy(crop_rgb)
                input_tensor = transform(pil_img)
                entry["tensor"] = input_tensor
                roi_entries.append(entry)

            # Build batch for valid ROIs
            valid_indices = [i for i, e in enumerate(roi_entries) if e["tensor"] is not None]
            probs_batch = None
            if valid_indices:
                batch_tensors = [roi_entries[i]["tensor"] for i in valid_indices]
                inputs = torch.stack(batch_tensors, dim=0).to(device)
                with torch.no_grad():
                    logits = model(inputs)
                    probs_batch = torch.softmax(logits, dim=1)

            current_status: Dict[str, Dict] = {}
            valid_ptr = 0  # index into probs_batch

            # Second pass: interpret predictions, update timers and draw overlays
            for idx, e in enumerate(roi_entries):
                sid = e["sid"]
                x1, y1, x2, y2 = e["x1"], e["y1"], e["x2"], e["y2"]

                if e["tensor"] is None:
                    # Invalid ROI
                    pred_class = "invalid"
                    conf_val = 0.0
                else:
                    # Get corresponding probs row from batch
                    assert probs_batch is not None
                    p = probs_batch[valid_ptr]
                    valid_ptr += 1
                    conf, pred_idx = torch.max(p, dim=0)
                    pred_class = idx_to_class[int(pred_idx)]
                    conf_val = float(conf.item())

                # previous status
                prev = slot_status.get(sid, {})
                prev_state = prev.get("status") if prev else None

                # Handle transitions for logging durations
                now_ts = time.time()
                last_duration = prev.get("last_duration", 0.0) if prev else 0.0
                total_occupied = prev.get("total_occupied", 0.0) if prev else 0.0
                purpose = prev.get("purpose") if prev else None
                expected_end = prev.get("expected_end") if prev else None

                if pred_class == "occupied":
                    # If it just became occupied, record start
                    if prev_state != "occupied":
                        slot_timers[sid] = now_ts
                        # clear prior last_duration
                        last_duration = 0.0
                        # mark awaiting purpose if none exists
                        if not purpose:
                            # frontend will prompt
                            pass
                    # compute ongoing duration
                    start_ts = slot_timers.get(sid, now_ts)
                    ongoing = now_ts - start_ts
                    last_duration = ongoing
                    # If there is an expected_end, compute remaining
                    if expected_end:
                        remaining = max(0.0, expected_end - now_ts)
                    else:
                        remaining = None
                elif pred_class == "empty":
                    # If it was occupied and now became empty, finalize duration
                    if prev_state == "occupied" and sid in slot_timers:
                        start_ts = slot_timers.pop(sid)
                        end_ts = now_ts
                        duration = end_ts - start_ts
                        last_duration = duration
                        total_occupied = (prev.get("total_occupied", 0.0) or 0.0) + duration
                        # append to CSV and in-memory log
                        log_entry = {
                            "timestamp": datetime.utcfromtimestamp(end_ts).isoformat() + "Z",
                            "slot_id": sid,
                            "duration_seconds": float(f"{duration:.3f}"),
                            "start_time": datetime.utcfromtimestamp(start_ts).isoformat() + "Z",
                            "end_time": datetime.utcfromtimestamp(end_ts).isoformat() + "Z",
                        }
                        occupancy_logs.append(log_entry)
                        append_log_csv(sid, start_ts, end_ts, duration)
                        purpose = None
                        expected_end = None
                    else:
                        # remain empty
                        last_duration = prev.get("last_duration", 0.0) if prev else 0.0
                        total_occupied = prev.get("total_occupied", 0.0) if prev else 0.0
                else:
                    # invalid or unknown
                    last_duration = prev.get("last_duration", 0.0) if prev else 0.0
                    total_occupied = prev.get("total_occupied", 0.0) if prev else 0.0

                # Update current status dict; do not overwrite purpose/expected_end unless set
                current_status[sid] = {
                    "status": pred_class,
                    "conf": conf_val,
                    "last_duration": float(last_duration),
                    "total_occupied": float(total_occupied),
                    "purpose": purpose,
                    "expected_end": expected_end,
                }

                # Draw overlay on display frame
                if pred_class == "empty":
                    color = (0, 255, 0)
                elif pred_class == "occupied":
                    color = (0, 0, 255)
                elif pred_class == "invalid":
                    color = (0, 180, 180)
                else:
                    color = (128, 128, 128)

                if not (x1 >= x2 or y1 >= y2):
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    label_text = f"{sid}:{pred_class} {conf_val:.2f}"
                    cv2.putText(display, label_text, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Atomically replace global status and latest frame
            slot_status = current_status
            ok, jpeg = cv2.imencode(".jpg", display)
            if ok:
                latest_frame_jpeg = jpeg.tobytes()

            time.sleep(poll_delay)
    except KeyboardInterrupt:
        print("\n[INF] Inference loop interrupted by user.")
    finally:
        cap.release()
        print("[INF] Inference loop stopped.")


# ---------- Flask app ----------

app = Flask(__name__)

# (DASHBOARD_HTML is identical to previous version with modal/front-end; omitted here for brevity)
# We'll reuse the earlier HTML. For brevity in this patch replace with full HTML used before.

DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Smart Parking Dashboard</title>
  <style>
    html,body{height:100%;margin:0;padding:0}
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0f172a; color: #e5e7eb; margin: 0; padding: 0; overflow-x:auto }
    header { padding: 12px 20px; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
    header h1 { margin: 0; font-size: 18px; }
    header span { font-size: 13px; color: #9ca3af; }
    .container { padding: 16px; }

    /* Layout: left = camera, right = slots panel */
    .main-flex { display: flex; gap: 18px; align-items: flex-start; flex-wrap: nowrap; }

    /* Camera box: fixed width and height so the right panel remains beside it */
    .camera-col { flex: 0 0 640px; }
    .camera-box { width: 640px; height: 480px; border-radius:12px; overflow:hidden; border:1px solid #1f2937; background:#020617; display:block }
    .camera-box img { display:block; width:100%; height:100%; object-fit:cover; }

    /* Slots panel: fixed width so it stays to the right of camera */
    .slots-panel { flex: 0 0 420px; width:420px; display: flex; flex-direction: column; gap: 12px; }

    /* Summary cards at top of right panel */
    .summary { display: flex; gap: 12px; margin-bottom: 8px; }
    .card { padding: 10px 12px; border-radius: 10px; background: #111827; border: 1px solid #1f2937; flex: 1; }
    .card h2 { margin: 0 0 4px 0; font-size: 13px; }
    .card p { margin: 0; font-size: 13px; color: #9ca3af; }

    /* Slots grid: 3 columns x 2 rows */
    .slots-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .slot { border-radius: 10px; padding: 10px; border: 1px solid #1f2937; background: linear-gradient(180deg,#051025,#021018); display: flex; flex-direction: column; gap: 6px; min-height: 92px }
    .slot-id { font-weight: 700; font-size: 15px; }
    .slot-status { font-size: 13px; font-weight: 700; }
    .slot-status.empty { color: #22c55e; }
    .slot-status.occupied { color: #f97373; }
    .slot-status.invalid { color: #facc15; }
    .slot-conf, .slot-duration, .slot-purpose { font-size: 12px; color: #9ca3af; white-space:nowrap; overflow:hidden; text-overflow:ellipsis }

    /* Logs below */
    .logs-box { margin-top: 10px; border-radius:10px; border:1px solid #122030; padding:8px; background:#081025; color:#9ca3af; max-height:180px; overflow:auto; }

    /* small screen behavior: stack vertically */
    @media (max-width: 1100px) {
      .main-flex { flex-direction: column; }
      .camera-col, .slots-panel { flex: none; width: 100%; }
      .camera-box { width:100%; height:360px }
      .slots-grid { grid-template-columns: repeat(2, 1fr); }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Smart Parking Dashboard</h1>
      <span>Real-time slot occupancy from camera feed</span>
    </div>
    <div>
      <span id="last-updated">Last update: --</span>
    </div>
  </header>

  <div class="container">
    <div class="main-flex">

      <!-- Left: Camera (fixed size column) -->
      <div class="camera-col">
        <div style="margin-bottom:8px;">
          <h2 style="font-size:16px; margin:0 0 6px 0; color:#e5e7eb;">Live Camera Preview</h2>
        </div>
        <div class="camera-box">
          <img id="camera-feed" src="/video_feed" alt="camera feed" />
        </div>
      </div>

      <!-- Right: Slots panel with summary, grid (3x2), logs -->
      <div class="slots-panel">
        <div class="summary">
          <div class="card"><h2>Empty</h2><p><span id="empty-count">0</span></p></div>
          <div class="card"><h2>Occupied</h2><p><span id="occupied-count">0</span></p></div>
          <div class="card"><h2>Total</h2><p><span id="total-count">0</span></p></div>
        </div>

        <div class="slots-grid" id="slots-grid" aria-live="polite">
          <!-- JS will populate slot boxes here (3 columns) -->
        </div>

        <div style="margin-top:6px;">
          <h3 style="margin:6px 0 8px 0;">Recent occupancy logs</h3>
          <div class="logs-box" id="logs"></div>
        </div>
      </div>

    </div>
  </div>

  <!-- Purpose modal (unchanged) -->
  <div id="purpose-modal" style="display:none;" class="modal">
    <div class="modal-box">
      <h3 id="modal-title">Set purpose</h3>
      <div><label for="purpose-select">Purpose:</label>
        <select id="purpose-select">
          <option value="shopping">Shopping (~90m)</option>
          <option value="eating">Eating (~60m)</option>
          <option value="cinema">Cinema (~180m)</option>
          <option value="work">Work (~240m)</option>
          <option value="other">Other (~30m)</option>
        </select>
      </div>
      <div style="margin-top:8px; text-align:right;">
        <button id="modal-cancel">Cancel</button>
        <button id="modal-save">Save</button>
      </div>
    </div>
  </div>

  <script>
    // Re-use existing fetchStatus/fetchLogs + rendering logic but target the new slots-grid
    let awaitingPurposeFor = null;
    async function fetchStatus() {
      try {
        const res = await fetch('/api/status'); if (!res.ok) return; const data = await res.json(); const slots = data.slot_status || {}; const grid = document.getElementById('slots-grid'); grid.innerHTML = '';
        let emptyCount=0, occCount=0; const ids = Object.keys(slots).sort();

        ids.forEach(id => {
          const info = slots[id]; const status = info.status||'unknown'; const conf = info.conf||0; const last_dur = info.last_duration||0; const total_occ = info.total_occupied||0; const purpose = info.purpose||null; const expected_end = info.expected_end||null;
          if (status==='empty') emptyCount++; if(status==='occupied') occCount++;

          const div = document.createElement('div'); div.className='slot';
          const idEl=document.createElement('div'); idEl.className='slot-id'; idEl.textContent=id;
          const statusEl=document.createElement('div'); statusEl.className='slot-status '+status; statusEl.textContent=status.toUpperCase();
          const confEl=document.createElement('div'); confEl.className='slot-conf'; confEl.textContent='Confidence: '+(conf*100).toFixed(1)+'%';
          const durEl=document.createElement('div'); durEl.className='slot-duration'; durEl.textContent='Last occ: '+(last_dur>0?(Math.round(last_dur)+'s'):'0s')+' | Total: '+Math.round(total_occ)+'s';
          const purposeEl=document.createElement('div'); purposeEl.className='slot-purpose';

          if (purpose) {
            if (info.predicted_duration != null) {
              const mean = Math.round(info.predicted_duration);
              const std = Math.round(info.predicted_std || 0);
              const confScore = Math.round((info.predicted_confidence || 0) * 100);
              const rem = Math.round(info.predicted_remaining || 0);
              const mins = Math.floor(rem / 60);
              const secs = Math.round(rem % 60);
              const remText = mins + 'm ' + secs + 's';
              purposeEl.textContent = `Purpose: ${purpose} | Pred: ${Math.round(mean/60)}m ± ${Math.round(std/60)}m | Remaining: ${remText} | Conf: ${confScore}%`;
            } else {
              let remText = '--';
              if (expected_end) {
                const rem = Math.max(0, Math.round(expected_end - (Date.now()/1000)));
                const mins = Math.floor(rem / 60);
                const secs = Math.round(rem % 60);
                remText = mins + 'm ' + secs + 's';
              }
              purposeEl.textContent = 'Purpose: ' + purpose + ' | Remaining: ' + remText;
            }
          } else {
            purposeEl.textContent = 'Purpose: -';
          }

          div.appendChild(idEl); div.appendChild(statusEl); div.appendChild(confEl); div.appendChild(durEl); div.appendChild(purposeEl);
          grid.appendChild(div);

          if(status==='occupied' && !purpose && !awaitingPurposeFor){awaitingPurposeFor=id; showPurposeModal(id);}        });

        document.getElementById('empty-count').textContent=emptyCount; document.getElementById('occupied-count').textContent=occCount; document.getElementById('total-count').textContent=ids.length; document.getElementById('last-updated').textContent='Last update: '+(new Date()).toLocaleTimeString();
      } catch(e){console.error(e);} }

    function showPurposeModal(slotId){const modal=document.getElementById('purpose-modal'); modal.style.display='flex'; document.getElementById('modal-title').textContent=`Set purpose for ${slotId}`; document.getElementById('modal-cancel').onclick=function(){modal.style.display='none'; awaitingPurposeFor=null}; document.getElementById('modal-save').onclick=async function(){const val=document.getElementById('purpose-select').value; try{await fetch('/api/set_purpose',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({slot_id:slotId,purpose:val})});}catch(e){console.error(e);} modal.style.display='none'; awaitingPurposeFor=null;}}

    async function fetchLogs(){try{const res=await fetch('/api/logs'); if(!res.ok) return; const data=await res.json(); const logs=data.logs||[]; const el=document.getElementById('logs'); el.innerHTML=''; logs.forEach(l=>{const row=document.createElement('div'); row.textContent=`${l.timestamp}  ${l.slot_id}  ${l.duration_seconds}s`; el.appendChild(row);});}catch(e){console.error(e);} }
    setInterval(()=>{fetchStatus();fetchLogs();},1000); window.onload=()=>{fetchStatus();fetchLogs();};
  </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    # Enrich slot_status with computed remaining seconds for frontend
    enriched = {}
    now_ts = time.time()
    predictor = load_predictor()

    for sid, info in slot_status.items():
        copy = dict(info)
        expected_end = info.get("expected_end")
        if expected_end:
            copy["expected_end"] = expected_end

        # If slot is occupied and has a purpose, try to compute model prediction + uncertainty
        try:
            if predictor is not None and copy.get("status") == "occupied" and copy.get("purpose"):
                # derive features
                start_hour = time.localtime(now_ts).tm_hour
                day_of_week = time.localtime(now_ts).tm_wday
                X = pd.DataFrame([{'purpose': copy.get('purpose'), 'start_hour': start_hour, 'day_of_week': day_of_week}])

                # use pipeline to transform then use RF estimators to compute per-tree predictions
                # if pipeline has named_steps['pre'] and named_steps['rf']
                pre = predictor.named_steps.get('pre') if hasattr(predictor, 'named_steps') else None
                rf = predictor.named_steps.get('rf') if hasattr(predictor, 'named_steps') else None

                if pre is not None and rf is not None and hasattr(rf, 'estimators_'):
                    X_trans = pre.transform(X)
                    preds = np.array([est.predict(X_trans) for est in rf.estimators_])
                    preds = preds.reshape(preds.shape[0], -1)
                    mean_pred = float(np.mean(preds))
                    std_pred = float(np.std(preds))
                    # simple confidence score: 1 - (std/mean) clipped to [0,1]
                    conf = 0.0
                    if mean_pred > 0:
                        conf = max(0.0, min(1.0, 1.0 - (std_pred / mean_pred)))
                    # compute elapsed since occupied start if available
                    start_ts = slot_timers.get(sid, now_ts)
                    elapsed = max(0.0, now_ts - start_ts)
                    predicted_remaining = max(0.0, mean_pred - elapsed)

                    copy['predicted_duration'] = mean_pred
                    copy['predicted_std'] = std_pred
                    copy['predicted_confidence'] = conf
                    copy['predicted_remaining'] = predicted_remaining
        except Exception:
            # if anything fails, skip adding predictions
            pass

        enriched[sid] = copy
    return jsonify(slot_status=enriched)
    # Enrich slot_status with computed remaining seconds for frontend
    enriched = {}
    now_ts = time.time()
    for sid, info in slot_status.items():
        copy = dict(info)
        expected_end = info.get("expected_end")
        if expected_end:
            copy["expected_end"] = expected_end
        enriched[sid] = copy
    return jsonify(slot_status=enriched)
# ---------- Purpose setting and predictor endpoints ----------

@app.route('/api/set_purpose', methods=['POST'])
def api_set_purpose():
    """Set purpose for a slot. Body: {slot_id, purpose}. This sets an expected_end time using the trained predictor if available, otherwise uses defaults."""
    body = request.get_json() or {}
    slot_id = body.get('slot_id')
    purpose = body.get('purpose')
    if not slot_id or not purpose:
        return jsonify({'ok': False, 'error': 'slot_id and purpose required'}), 400

    now_ts = time.time()
    # default expected end using rule
    minutes = PURPOSE_DEFAULT_MINUTES.get(purpose, PURPOSE_DEFAULT_MINUTES['other'])
    expected_end = now_ts + minutes * 60

    # try predictor if exists
    try:
        # derive start_hour/day_of_week
        start_hour = time.localtime(now_ts).tm_hour
        day_of_week = time.localtime(now_ts).tm_wday
        pred = predict_duration_with_model(purpose, start_hour, day_of_week)
        if pred is not None:
            expected_end = now_ts + pred
    except Exception as e:
        print('[WARN] predictor failed:', e)

    # update global slot_status
    s = slot_status.get(slot_id, {})
    s['purpose'] = purpose
    s['expected_end'] = expected_end
    slot_status[slot_id] = s

    return jsonify({'ok': True, 'slot_id': slot_id, 'purpose': purpose, 'expected_end': expected_end})


@app.route('/api/generate_synthetic', methods=['POST'])
def api_generate_synthetic():
    """Generate a synthetic CSV dataset for predictor training. Returns the path."""
    count = int(request.args.get('n', 1000))
    # lightweight generator - reimplement here to ensure availability
    import random
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(SYNTH_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['slot_id','purpose','start_hour','day_of_week','duration_seconds'])
        slot_ids = [f'S{i:02d}' for i in range(1,21)]
        purposes = list(PURPOSE_DEFAULT_MINUTES.keys())
        for _ in range(count):
            sid = random.choice(slot_ids)
            p = random.choice(purposes)
            start_hour = random.randint(7,23)
            day = random.randint(0,6)
            base_minutes = PURPOSE_DEFAULT_MINUTES[p]
            duration_min = max(1, int(np.random.normal(loc=base_minutes, scale=base_minutes*0.4)))
            duration = duration_min * 60
            w.writerow([sid,p,start_hour,day,int(duration)])
    return jsonify({'ok': True, 'path': SYNTH_CSV})


@app.route('/api/train_predictor', methods=['POST'])
def api_train_predictor():
    """Train the time-to-empty predictor from synthetic + real logs.
    POST body optional param: n_samples for synthetic gen (if you want to force regenerate)
    """
    try:
        # optionally regenerate synthetic data
        n = int(request.args.get('n', 0))
        if n > 0:
            # call synthetic generator
            requests_count = n
            # reuse generator logic
            from flask import current_app
            # generate synthetic dataset
            import random
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(SYNTH_CSV, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['slot_id','purpose','start_hour','day_of_week','duration_seconds'])
                slot_ids = [f'S{i:02d}' for i in range(1,21)]
                purposes = list(PURPOSE_DEFAULT_MINUTES.keys())
                for _ in range(n):
                    sid = random.choice(slot_ids)
                    p = random.choice(purposes)
                    start_hour = random.randint(7,23)
                    day = random.randint(0,6)
                    base_minutes = PURPOSE_DEFAULT_MINUTES[p]
                    duration_min = max(1, int(np.random.normal(loc=base_minutes, scale=base_minutes*0.4)))
                    duration = duration_min * 60
                    w.writerow([sid,p,start_hour,day,int(duration)])
        path, score = train_time_predictor()
        return jsonify({'ok': True, 'path': path, 'val_score': score})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route("/api/logs")
def api_logs():
    # Return recent in-memory logs (last 50)
    recent = occupancy_logs[-50:][::-1]
    return jsonify(logs=recent)


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the latest annotated camera frame."""
    def generate():
        global latest_frame_jpeg
        while True:
            frame = latest_frame_jpeg
            if frame is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.01)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------- Entry point ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Flask host")
    parser.add_argument("--port", type=int, default=8000, help="Flask port")
    args = parser.parse_args()

    # Start inference loop in background thread
    t = threading.Thread(target=inference_loop, kwargs={"camera_index": args.camera}, daemon=True)
    t.start()

    print(f"[WEB] Starting dashboard on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()