from pathlib import Path

# Canonical project structure
structure = {
    "data/enroll": [],
    "data/db": [],
    "models": [
        "embedder_arcface.onnx",
    ],
    "src": [
        "camera.py",
        "detect.py",
        "landmarks.py",
        "align.py",
        "embed.py",
        "enroll.py",
        "recognize.py",
        "evaluate.py",
        "haar_5pt.py",
    ],
    "book": [],
}

for folder, files in structure.items():
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD

=======
>>>>>>> bbdd871d1afc040d42a68aa60abbfcccf7562b1f
    for file in files:
        file_path = folder_path / file
        if not file_path.exists():
            file_path.touch()

<<<<<<< HEAD
print("face-recognition-5pt project structure created successfully.")
=======
print("face-recognition-5pt project structure created successfully.")
>>>>>>> bbdd871d1afc040d42a68aa60abbfcccf7562b1f
