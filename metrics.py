import numpy as np
import os

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


def compute_f1(pred_boxes, gt_boxes, iou_thresh=0.5):
    tp, fp = 0, 0
    matched = set()

    for p in pred_boxes:
        found = False
        for i, g in enumerate(gt_boxes):
            if i in matched:
                continue
            if compute_iou(p, g) > iou_thresh:
                tp += 1
                matched.add(i)
                found = True
                break
        if not found:
            fp += 1

    fn = len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return 2 * precision * recall / (precision + recall + 1e-6)


def compute_avg_iou(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    ious = []
    for p in pred_boxes:
        best = 0
        for g in gt_boxes:
            best = max(best, compute_iou(p, g))
        ious.append(best)

    return np.mean(ious)


def compute_detection_score(pred_boxes, gt_boxes, gamma=0.1):
    f1 = compute_f1(pred_boxes, gt_boxes)
    iou = compute_avg_iou(pred_boxes, gt_boxes)
    return gamma * iou + (1 - gamma) * f1


# 🔥 FIXED: convert normalized → pixel
def load_gt_boxes(label_path, img_shape):
    boxes = []

    if not os.path.exists(label_path):
        return boxes

    h, w = img_shape[:2]

    with open(label_path) as f:
        for line in f:
            _, x, y, bw, bh = map(float, line.strip().split())

            x1 = (x - bw/2) * w
            y1 = (y - bh/2) * h
            x2 = (x + bw/2) * w
            y2 = (y + bh/2) * h

            boxes.append([x1, y1, x2, y2])

    return boxes