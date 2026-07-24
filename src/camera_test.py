import cv2
import time

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open failed")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    now = time.time()
    fps = 1/(now-prev)
    prev = now

    cv2.putText(frame,
                f"FPS : {fps:.1f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
