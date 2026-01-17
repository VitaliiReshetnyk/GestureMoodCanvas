# GestureMoodCanvas

GestureMoodCanvas is an interactive computer vision prototype that demonstrates
real-time human–computer interaction using hand gestures, facial expressions,
and body pose.

The project is implemented in Python using MediaPipe Tasks and OpenCV and is
designed as a modular MVP suitable for further extension.

---

## Features

- Hand gesture–based drawing on a virtual canvas
- Facial expression detection with visual overlays
- Body pose detection for global actions
- Fully on-device inference
- Modular and extensible architecture

---

## Tech Stack

- Python 3.11
- MediaPipe Tasks (Hand, Face, Pose Landmarkers)
- OpenCV
- NumPy

---

## Controls

### Hand Gestures
- Index finger up — draw
- Fist — stop drawing
- Pinch (thumb + index) — toggle eraser
- Open palm — switch color

### Body Pose
- Both hands raised above shoulders — clear canvas

### Facial Expressions
- HAPPY, SURPRISED, ANGRY trigger visual overlays above the face

---

## Getting Started

### Install dependencies
```bash
pip install -r requirements.txt
