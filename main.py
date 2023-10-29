import cv2
from ultralytics import YOLO

cap = cv2.VideoCapture("dogs.mp4")

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()