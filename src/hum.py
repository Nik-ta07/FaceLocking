import cv2

for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    print(f"Camera {i}: {ok}")
    if ok:
        ret, frame = cap.read()
        print("   frame:", ret)
    cap.release()
