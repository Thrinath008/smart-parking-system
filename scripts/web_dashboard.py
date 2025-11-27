
"""
web_dashboard.py

Run real-time parking slot inference in the background and expose:

- A web dashboard at `/` (HTML+JS) to visualize slot status
- A JSON API at `/api/status` with current slot predictions
- A JSON API at `/api/logs` returning recent occupancy logs

Usage:
  python scripts/web_dashboard.py --camera 0

Then open: http://localhost:8000 in your browser.
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

from flask import Flask, jsonify, render_template_string, Response


ROIS_PATH = "data/processed/rois.json"
MODEL_PATH = "models/trained/slot_classifier_best.pth"
LOG_DIR = "data/logs"
LOG_CSV = os.path.join(LOG_DIR, "occupancy_log.csv")

# Shared state between inference loop and Flask
slot_status: Dict[str, Dict] = {}  # e.g. {"A1": {"status": "empty", "conf": 0.98, "last_duration": 0.0, "total_occupied": 0.0}}
latest_frame_jpeg = None  # type: ignore  # most recent annotated frame as JPEG bytes

# Internal tracking: start time per slot when it became occupied
slot_timers: Dict[str, float] = {}
# Simple in-memory log buffer (also persisted to CSV)
occupancy_logs: List[Dict[str, Any]] = []


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
    # Force CPU for smoother, more stable webcam + UI behavior on macOS
    return torch.device("cpu")


def build_transform(img_size: int):
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


# ---------- Inference loop (background thread) ----------

def inference_loop(camera_index: int = 0, poll_delay: float = 0.05):
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
            current_status: Dict[str, Dict] = {}

            for r in rois:
                sid = r["id"]
                x, y, w, h = r["bbox"]

                # clip bbox within frame
                x2 = max(0, min(x + w, w0))
                y2 = max(0, min(y + h, h0))
                x1 = max(0, min(x, w0 - 1))
                y1 = max(0, min(y, h0 - 1))

                if x1 >= x2 or y1 >= y2:
                    current_status[sid] = {"status": "invalid", "conf": 0.0, "last_duration": 0.0, "total_occupied": 0.0}
                    continue

                crop = frame[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pil_img = pil_from_numpy(crop_rgb)
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    conf, pred_idx = torch.max(probs, dim=0)
                    pred_class = idx_to_class[int(pred_idx)]
                    conf_val = float(conf.item())

                # previous status
                prev = slot_status.get(sid, {})
                prev_state = prev.get("status") if prev else None

                # Handle transitions for logging durations
                now_ts = time.time()
                last_duration = prev.get("last_duration", 0.0) if prev else 0.0
                total_occupied = prev.get("total_occupied", 0.0) if prev else 0.0

                if pred_class == "occupied":
                    # If it just became occupied, record start
                    if prev_state != "occupied":
                        slot_timers[sid] = now_ts
                    # compute ongoing duration
                    start_ts = slot_timers.get(sid, now_ts)
                    ongoing = now_ts - start_ts
                    last_duration = ongoing
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
                    else:
                        # remain empty
                        last_duration = prev.get("last_duration", 0.0) if prev else 0.0
                        total_occupied = prev.get("total_occupied", 0.0) if prev else 0.0
                else:
                    # invalid or unknown
                    last_duration = prev.get("last_duration", 0.0) if prev else 0.0
                    total_occupied = prev.get("total_occupied", 0.0) if prev else 0.0

                current_status[sid] = {
                    "status": pred_class,
                    "conf": conf_val,
                    "last_duration": float(last_duration),
                    "total_occupied": float(total_occupied),
                }

                # Draw overlay on display frame
                if pred_class == "empty":
                    color = (0, 255, 0)
                elif pred_class == "occupied":
                    color = (0, 0, 255)
                else:
                    color = (0, 180, 180)

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

DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Smart Parking Dashboard</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      margin: 0;
      padding: 0;
    }
    header {
      padding: 16px 24px;
      border-bottom: 1px solid #1f2937;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    header h1 {
      margin: 0;
      font-size: 20px;
    }
    header span {
      font-size: 13px;
      color: #9ca3af;
    }
    main {
      padding: 24px;
    }
    .summary {
      display: flex;
      gap: 16px;
      margin-bottom: 24px;
    }
    .card {
      padding: 16px 20px;
      border-radius: 12px;
      background: #111827;
      border: 1px solid #1f2937;
      flex: 1;
    }
    .card h2 {
      margin: 0 0 4px 0;
      font-size: 16px;
    }
    .card p {
      margin: 0;
      font-size: 14px;
      color: #9ca3af;
    }
    .slots-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
      gap: 12px;
    }
    .slot {
      border-radius: 12px;
      padding: 12px;
      border: 1px solid #1f2937;
      background: #020617;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }
    .slot-id {
      font-weight: 600;
      font-size: 14px;
    }
    .slot-status {
      font-size: 13px;
      font-weight: 500;
    }
    .slot-status.empty {
      color: #22c55e;
    }
    .slot-status.occupied {
      color: #f97373;
    }
    .slot-status.invalid {
      color: #facc15;
    }
    .slot-conf {
      font-size: 12px;
      color: #9ca3af;
    }
    .slot-duration {
      font-size: 12px;
      color: #9ca3af;
    }
    .pill {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      margin-left: 6px;
      background: #1f2937;
      color: #9ca3af;
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
  <main>
    <div class="summary">
      <div class="card">
        <h2>Empty slots</h2>
        <p><span id="empty-count">0</span></p>
      </div>
      <div class="card">
        <h2>Occupied slots</h2>
        <p><span id="occupied-count">0</span></p>
      </div>
      <div class="card">
        <h2>Total slots</h2>
        <p><span id="total-count">0</span></p>
      </div>
    </div>
    <div style="margin-bottom:24px;">
      <h2 style="font-size:16px; margin:0 0 8px 0;">Live Camera Preview</h2>
      <div style="border-radius:12px; overflow:hidden; border:1px solid #1f2937; max-width:640px;">
        <img id="camera-feed" src="/video_feed" style="display:block; width:100%; height:auto;" />
      </div>
    </div>
    <div class="slots-grid" id="slots-grid">
      <!-- Filled by JS -->
    </div>
    <div style="margin-top:20px;">
      <h3 style="margin:0 0 8px 0;">Recent occupancy logs</h3>
      <div id="logs" style="font-size:13px; color:#9ca3af; max-height:180px; overflow:auto; background:#081025; padding:8px; border-radius:8px; border:1px solid #122030;"></div>
    </div>
  </main>

  <script>
    async function fetchStatus() {
      try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const data = await res.json();
        const slots = data.slot_status || {};
        const grid = document.getElementById('slots-grid');
        grid.innerHTML = '';

        let emptyCount = 0;
        let occCount = 0;
        const ids = Object.keys(slots).sort();

        ids.forEach(id => {
          const info = slots[id];
          const status = info.status || 'unknown';
          const conf = info.conf != null ? info.conf : 0.0;
          const last_dur = info.last_duration != null ? info.last_duration : 0.0;
          const total_occ = info.total_occupied != null ? info.total_occupied : 0.0;

          if (status === 'empty') emptyCount++;
          if (status === 'occupied') occCount++;

          const div = document.createElement('div');
          div.className = 'slot';

          const idEl = document.createElement('div');
          idEl.className = 'slot-id';
          idEl.textContent = id;

          const statusEl = document.createElement('div');
          statusEl.className = 'slot-status ' + status;
          statusEl.textContent = status.toUpperCase();

          const confEl = document.createElement('div');
          confEl.className = 'slot-conf';
          confEl.textContent = 'Confidence: ' + (conf * 100).toFixed(1) + '%';

          const durEl = document.createElement('div');
          durEl.className = 'slot-duration';
          durEl.textContent = 'Last occ: ' + (last_dur > 0 ? (Math.round(last_dur) + 's') : '--') + ' | Total: ' + Math.round(total_occ) + 's';

          div.appendChild(idEl);
          div.appendChild(statusEl);
          div.appendChild(confEl);
          div.appendChild(durEl);
          grid.appendChild(div);
        });

        document.getElementById('empty-count').textContent = emptyCount;
        document.getElementById('occupied-count').textContent = occCount;
        document.getElementById('total-count').textContent = ids.length;

        const now = new Date();
        document.getElementById('last-updated').textContent =
          'Last update: ' + now.toLocaleTimeString();
      } catch (e) {
        console.error(e);
      }
    }

    async function fetchLogs() {
      try {
        const res = await fetch('/api/logs');
        if (!res.ok) return;
        const data = await res.json();
        const logs = data.logs || [];
        const el = document.getElementById('logs');
        el.innerHTML = '';
        logs.forEach(l => {
          const row = document.createElement('div');
          row.textContent = `${l.timestamp}  ${l.slot_id}  ${l.duration_seconds}s`;
          el.appendChild(row);
        });
      } catch (e) {
        console.error(e);
      }
    }

    setInterval(() => { fetchStatus(); fetchLogs(); }, 1000);
    window.onload = () => { fetchStatus(); fetchLogs(); };
  </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    # Return the latest slot_status dict
    return jsonify(slot_status=slot_status)


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