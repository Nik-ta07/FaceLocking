import cv2
from .camera_config import WEBCAM_INDEX, WEBCAM_BACKEND

cap = cv2.VideoCapture(WEBCAM_INDEX, WEBCAM_BACKEND)


if not cap.isOpened():
    print("❌ Error: Could not open camera")
    exit()

ret, frame = cap.read()
if not ret:
    print("❌ Error: Could not read frame")
    exit()

print("✅ Camera is working!")
cv2.imwrite("test_photo.jpg", frame)
print("📸 Test photo saved as 'test_photo.jpg'")
cap.release()