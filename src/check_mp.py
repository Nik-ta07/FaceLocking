try:
    import mediapipe
    print(f"MediaPipe found at: {mediapipe.__file__}")
    print(f"MediaPipe version: {getattr(mediapipe, '__version__', 'unknown')}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
