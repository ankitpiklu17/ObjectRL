from utils.dataset import VOCDataset
from detector.yolo_detector import YOLODetector
from utils.metrics import load_gt_boxes

dataset = VOCDataset("/home/ankit/datasets/VOC/images/train2007")
detector = YOLODetector()

img, path = dataset.sample()

pred_boxes, _ = detector.predict(img)

label_path = path.replace("images", "labels").replace(".jpg", ".txt")
gt_boxes = load_gt_boxes(label_path)

print("Pred:", len(pred_boxes))
print("GT:", len(gt_boxes))