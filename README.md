# Face Locking & Recognition System (Windows)

This project implements a **real-time face recognition and face locking system** using **5-point facial landmarks**, **face alignment**, and **ArcFace embeddings**.

**Project URL**: [https://github.com/Nik-ta07/FaceLocking.git](https://github.com/Nik-ta07/FaceLocking.git)

The system allows you to:
1.  **Enroll** new faces.
2.  **Recognize** faces in real-time.
3.  **Lock** onto a specific target.
4.  **Detect** actions (blinks, smiles, head movement) and record history.

---

## � 1. Setup & Installation

### Clone the Repository
```powershell
git clone https://github.com/Nik-ta07/FaceLocking.git
cd FaceLocking
```

### Create & Activate Virtual Environment
```powershell
# Create venv
python -m venv .venv

# Activate venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies
```powershell
pip install --upgrade pip
pip install opencv-python numpy onnxruntime mediapipe insightface

# Fix for MediaPipe on Windows (if needed)
pip uninstall mediapipe -y
pip install mediapipe==0.10.9
```

### Fix ArcFace Model
Ensure the ONNX model is in the right place:
```powershell
Copy-Item buffalo_l\w600k_r50.onnx models\embedder_arcface.onnx
```

---

## 🎮 2. Usage Instructions

Follow these steps in order to use the system.

### Step 1: Test Camera
Ensure your webcam is working correctly.
```powershell
python -m src.testcamera
```

### Step 2: Camera Configuration
Configure camera settings if necessary.
```powershell
python -m src.camera
```

### Step 3: Enroll Faces
Register new identities into the database.
```powershell
python -m src.enroll
```
*Follow on-screen instructions to capture face data.*

### Step 4: Recognize Faces
Test the recognition system to verify enrolled faces are detected.
```powershell
python -m src.recognise
```

### Step 5: Face Locking
Run the specialized locking module to track a specific person.
```powershell
python -m src.lock
```

### Step 6: Full Detection System
Run the main detection system that combines everything.
```powershell
python -m src.detect
```

---

## 📁 Project Structure

```text
FaceLocking/
├── .venv/                  # Virtual Environment
├── data/
│   ├── enroll/             # Enrolled face images
│   └── db/                 # Database files (.npz, .json)
├── models/
│   └── embedder_arcface.onnx
├── src/
│   ├── detect.py           # Main detection logic
│   ├── lock.py             # Locking logic
│   ├── enroll.py           # Enrollment script
│   ├── recognize.py        # Recognition script
│   └── ...
├── init_project.py
└── README.md
```

# 🐍 Python Version
Python 3.11

## ❗ Troubleshooting

-   **ModuleNotFoundError**: Always run scripts as modules (e.g., `python -m src.filename`) from the root directory.
-   **Camera not opening**: Check if another app is using the camera or run `src.testcamera`.
