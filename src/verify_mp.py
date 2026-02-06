import sys
try:
    import mediapipe as mp
    with open("mp_status.txt", "w") as f:
        f.write(f"Location: {mp.__file__}\n")
        try:
             f.write(f"Solutions: {mp.solutions}\n")
        except AttributeError:
             f.write("Solutions: AttributeError\n")
except Exception as e:
    with open("mp_status.txt", "w") as f:
        f.write(f"Failed import: {e}")
