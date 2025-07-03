import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model=YOLO("yolov8n-pose.pt")


while True:
    success,frame = cap.read()
    if not success:
        break

    results=model(frame)
    frame=results[0].plot()

    cv2.imshow("frame",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows


