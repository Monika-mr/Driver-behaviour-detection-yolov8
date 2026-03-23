from ultralytics import YOLO
model = YOLO("best.pt")
model.predict("test.jpg", show=True)
