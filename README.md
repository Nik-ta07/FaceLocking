# 🔍 Face Recognition with ArcFace (ONNX) & 5-Point Alignment

A **CPU-only, research-grade face recognition system** built with **ArcFace embeddings** and **5-point facial landmark alignment**, designed for **clarity, robustness, and real-world deployment** on machines without GPU acceleration.

> Accurate. Modular. Reproducible. No CUDA required.

---

## ✨ Features

- ⚙️ **CPU-Only Inference** – runs smoothly on laptops and low-resource machines  
- 🧠 **ArcFace (ONNX, ResNet-50)** – 512-dimensional L2-normalized embeddings  
- 📐 **5-Point Face Alignment** – similarity transform to canonical 112×112 faces  
- 🎥 **Real-Time Recognition** – multi-face detection with temporal smoothing  
- 🔓 **Open-Set Recognition** – automatically rejects unknown identities  
- 📊 **Threshold Evaluation** – FAR / FRR based decision tuning  
- 🧩 **Modular Pipeline** – each stage testable independently  

---

## 🖥️ System Requirements

| Component | Requirement |
|---------|-------------|
| Python | 3.9+ (tested on 3.11) |
| OS | Windows / macOS / Linux |
| Camera | Webcam |
| RAM | ≥ 2 GB |
| GPU | ❌ Not required |

Check Python version:

```bash
python --version
1️⃣ Clone Repository
git clone https://github.com/Nik-ta07/-Face-Recog-arc-onnx.git
cd -Face-Recog-arc-onnx

2️⃣ Create Virtual Environment
python3.11 -m venv .venv


Activate:

Windows (PowerShell)

.venv\Scripts\Activate.ps1


macOS / Linux

source .venv/bin/activate

3️⃣ Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

🧠 ArcFace Model Setup

Download the official InsightFace ArcFace ONNX model:

curl -L -o buffalo_l.zip \
https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download

unzip buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm buffalo_l.zip *.onnx

📁 Project Structure
Face-Recog-arc-onnx/
│
├── src/
│   ├── camera.py        # Camera validation
│   ├── detect.py        # Haar face detection
│   ├── landmarks.py     # 5-point landmark extraction
│   ├── align.py         # 112×112 face alignment
│   ├── embed.py         # ArcFace embedding extraction
│   ├── enroll.py        # Identity enrollment
│   ├── evaluate.py      # FAR / FRR threshold evaluation
│   └── recognize.py    # Live face recognition
│
├── data/
│   ├── enroll/          # Aligned enrollment images
│   └── db/              # Face database (NPZ + JSON)
│
├── models/
│   └── embedder_arcface.onnx
│
├── requirements.txt
└── README.md

🚀 Quick Start

Test each module independently:

python -m src.camera
python -m src.detect
python -m src.landmarks
python -m src.align
python -m src.embed


Enroll identities and start recognition:

python -m src.enroll
python -m src.evaluate
python -m src.recognize

🔄 Pipeline Overview
Enrollment Pipeline
Camera
 → Face Detection
 → 5-Point Landmarks
 → Alignment (112×112)
 → ArcFace Embedding
 → L2 Normalization
 → Mean Template
 → Database Storage

Recognition Pipeline
Camera
 → Detection + Alignment
 → ArcFace Embedding
 → Cosine Distance Matching
 → Threshold Decision
 → Identity / Unknown
 