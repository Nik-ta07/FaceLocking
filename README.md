Face Recognition System using ArcFace (ONNX) with 5-Point Face Alignment

A CPU-optimized, research-quality face recognition pipeline that combines classical detection with modern deep embeddings. Designed for clarity, reproducibility, and real-world deployment on machines without GPU acceleration.

Inspired by academic best practices and built around ArcFace embeddings with 5-point landmark alignment.

Overview

This project implements a complete face recognition pipeline:

Detect faces from a live camera stream

Extract 5 facial landmarks

Align faces into a canonical pose (112×112)

Generate 512-D ArcFace embeddings using ONNX Runtime

Perform open-set recognition via cosine distance

Persist identities using a lightweight on-disk database

Everything runs entirely on CPU, making it suitable for laptops, classrooms, and field deployments.

Highlights

🧠 ArcFace ONNX (ResNet-50)
High-quality 512-dimensional embeddings with L2 normalization

🖥️ CPU-Only Inference
No CUDA, no GPU dependencies, reproducible across machines

📐 5-Point Face Alignment
Haar detection + MediaPipe landmarks → similarity transform

🎥 Real-Time Multi-Face Recognition
Stable tracking with temporal smoothing

🔓 Open-Set Recognition
Automatically rejects unknown identities

📊 Threshold Evaluation Tool
FAR / FRR analysis for principled decision boundaries

🧩 Modular Architecture
Each stage runnable and testable independently

System Requirements
Component	Requirement
Python	3.9+ (tested on 3.11)
OS	Windows / macOS / Linux
Camera	USB or built-in webcam
RAM	≥ 2 GB
Storage	~500 MB

Verify Python version:

python --version

Installation
1. Clone the Repository
git clone https://github.com/Nik-ta07/-Face-Recog-arc-onnx.git
cd Face-Recog-onnx

2. Create & Activate Virtual Environment
python3.11 -m venv .venv


macOS / Linux
cd
source .venv/bin/activate


Windows (PowerShell)

.venv\Scripts\Activate.ps1

3. Install Dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt


Or manually:

pip install numpy opencv-python onnxruntime scipy mediapipe tqdm protobuf

ArcFace Model Setup

Download the official InsightFace ArcFace ONNX model:

curl -L -o buffalo_l.zip \
https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download


Extract and keep only the embedding model:

unzip buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm buffalo_l.zip *.onnx

Project Layout
Face-Recog-onnx/
│
├── src/
│   ├── camera.py        # Camera sanity check
│   ├── detect.py        # Haar face detection
│   ├── landmarks.py     # 5-point landmark extraction
│   ├── align.py         # 112×112 face alignment
│   ├── embed.py         # ArcFace ONNX inference
│   ├── enroll.py        # Identity enrollment
│   ├── evaluate.py      # FAR / FRR threshold analysis
│   ├── recognize.py    # Live recognition pipeline
│   └── haar_5pt.py      # Unified detector class
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

Quick Start

Run each component independently to verify functionality:

python -m src.camera
python -m src.detect
python -m src.landmarks
python -m src.align
python -m src.embed


Then:

python -m src.enroll
python -m src.evaluate
python -m src.recognize

Core Workflow
Enrollment Pipeline
Camera Frame
 → Face Detection
 → 5-Point Landmarks
 → Alignment (112×112)
 → ArcFace Embedding
 → L2 Normalization
 → Mean Template
 → Database Storage

Recognition Pipeline
Camera Frame
 → Detection + Alignment
 → Embedding Extraction
 → Cosine Distance Matching
 → Threshold Decision
 → Identity / Unknown

Threshold Evaluation

The evaluation tool computes genuine vs impostor distributions:

Genuine = same person

Impostor = different people

Output includes:

Mean / std / percentiles

FAR / FRR sweep

Suggested operating threshold

Typical output:

Suggested threshold (FAR ≈ 1%): dist ≈ 0.34
Equivalent cosine similarity ≈ 0.66


This value should be set in recognize.py.

Controls (Live Recognition)
Key	Action
q	Quit
r	Reload database
+	Increase threshold
-	Decrease threshold
d	Debug overlay
Database Format
face_db.npz

Maps identity → 512-D embedding

All vectors are L2-normalized

face_db.json

Metadata (names, timestamps, sample counts)

Common Issues

Camera not opening

Try changing camera index

Check OS camera permissions

MediaPipe import error

pip install mediapipe==0.10.32


Model not found

Confirm models/embedder_arcface.onnx exists

Low accuracy

Enroll more samples (20–30 recommended)

Improve lighting consistency

Re-run threshold evaluation

Visually verify alignment output

Design Notes

All embeddings have unit norm

Cosine similarity = dot product

Distance = 1 - similarity

CPU-only execution ensures deterministic behavior

Each pipeline stage is replaceable

References

Deng et al., ArcFace: Additive Angular Margin Loss, CVPR 2019

InsightFace Project

MediaPipe Face Mesh

ONNX Runtime

License

Educational and research use.
Derived from academic face recognition coursework.