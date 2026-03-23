# Driver-behaviour-detection-yolov8
YOLOv8 based driver behaviour detection with real-time inference

# infer_webcam.py
import cv2
from ultralytics import YOLO
model = YOLO("runs/detect/train3/weights/best.pt")
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    annotated = results[0].plot()
    cv2.imshow("Driver Monitoring", annotated)
    # ⭐ CLEAN EXIT
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

# infer_image.py
from ultralytics import YOLO
model = YOLO(r"C:\Users\mrmon\Downloads\Annotation\annotation 2\Monika\monica.v1i.yolov5pytorch\runs\detect\train3\weights\best.pt")
model.predict(r"C:\Users\mrmon\Downloads\Annotation\annotation 2\Monika\monica.v1i.yolov5pytorch\train\images\frame_1007_jpg.rf.1585c669db0dec7b527ebfcfb1cff116.jpg", show=True)
