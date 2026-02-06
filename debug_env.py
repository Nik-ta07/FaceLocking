import sys
import os
import importlib.util

print(f"Python Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print(f"System Path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to find mediapipe spec...")
try:
    spec = importlib.util.find_spec("mediapipe")
    print(f"Spec: {spec}")
    if spec:
        print(f"Origin: {spec.origin}")
except Exception as e:
    print(f"Error finding spec: {e}")

print("\nTrying import...")
try:
    import mediapipe
    print(f"Import successful: {mediapipe}")
    print(f"File: {mediapipe.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Other Error: {e}")
