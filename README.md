# Smart Parking Occupancy Detection (CV + Deep Learning)

Smart Parking Occupancy Detection is a real-time computer vision system that turns a simple printed parking layout + any camera into an **AI-powered parking monitor**.

Using OpenCV and a lightweight CNN (MobileNetV3), the system:

- Lets you **define parking slots** directly from the camera feed
- **Collects training data** (cropped slot images labeled `empty` / `occupied`)
- **Trains a deep learning model** to classify each slot
- Runs **real-time inference** to draw green/red boxes and count available slots

> Designed for hackathons and quick prototyping — no external sensors, just a camera and a laptop.

---

## ✨ Features

- 🎯 **ROI-based slot definition**  
  Define parking slots visually by drawing rectangles on the live camera feed.

- 📸 **Automated data collection**  
  Collect labeled images per slot (`empty`, `occupied`) in batches with a single key press.

- 🧠 **Deep learning classifier (MobileNetV3-Small)**  
  Transfer learning on your own dataset for robust slot classification.

- 📡 **Real-time occupancy detection**  
  Per-slot predictions overlaid on the live camera stream (green = empty, red = occupied).

- 💾 **Modular pipeline**  
  Clean separation between:
  - data collection
  - training
  - inference

---

## 📂 Project Structure

```bash
parking-project/
├── data/
│   ├── processed/
│   │   └── rois.json          # Saved ROI definitions (parking slot coordinates)
│   └── raw/
│       ├── A1/
│       │   ├── empty/         # Cropped images when slot A1 is empty
│       │   └── occupied/      # Cropped images when slot A1 is occupied
│       ├── A2/
│       └── ...                # Similarly for A2, A3, ..., B4
│
├── models/
│   ├── trained/
│   │   └── slot_classifier_best.pth   # Trained PyTorch model
│   └── onnx/                          # (Optional) Exported ONNX models
│
├── notebooks/
│   └── training_experiments.ipynb     # (Optional) Experiment playground
│
├── scripts/
│   ├── collect_data.py        # ROI definition + data collection
│   ├── train_model.py         # CNN training pipeline (MobileNetV3)
│   └── realtime_inference.py  # Real-time camera inference
│
├── requirements.txt
└── README.md