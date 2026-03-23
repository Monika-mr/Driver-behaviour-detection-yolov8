from ultralytics import YOLO
model = YOLO(r"C:\Users\mrmon\Downloads\Annotation\annotation 2\Monika\monica.v1i.yolov5pytorch\runs\detect\train3\weights\best.pt")
model.predict(r"C:\Users\mrmon\Downloads\Annotation\annotation 2\Monika\monica.v1i.yolov5pytorch\train\images\frame_1007_jpg.rf.1585c669db0dec7b527ebfcfb1cff116.jpg", show=True)
