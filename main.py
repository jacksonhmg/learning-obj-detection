# import cv2
# from ultralytics import YOLO
# import numpy as np

# cap = cv2.VideoCapture("dogs.mp4")

# model = YOLO("yolov8m.pt")


# while True:
#     ret, frame = cap.read()
#     if not ret: 
#         break

#     results = model(frame)
#     result = results[0]
#     print("box is", result.boxes)
#     bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
#     classes = np.array(result.boxes.cls.cpu(), dtype="int")
#     for cls, bbox in zip(classes, bboxes):
#         (x1, y1, x2, y2) = bbox

#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#         cv2.putText(frame, str(cls), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


#     cv2.imshow("Img", frame)
#     key = cv2.waitKey(0)
#     if key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()

from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model(source=1, show=True, conf=0.4, save=True)