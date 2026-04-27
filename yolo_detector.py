from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def predict(self, img):
        results = self.model(img, verbose=False)[0]

        boxes = []
        scores = []

        if results.boxes is None:
            return boxes, scores

        for b in results.boxes:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
            scores.append(float(b.conf[0]))

        return boxes, scores