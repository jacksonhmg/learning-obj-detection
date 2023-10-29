import cv2
from ultralytics import YOLO
import numpy as np

cap = cv2.VideoCapture("dogs.mp4")

model = YOLO("yolov8m.pt")


while True:
    ret, frame = cap.read()
    if not ret: 
        break

    results = model(frame)
    result = results[0]
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)


    cv2.imshow("Img", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()