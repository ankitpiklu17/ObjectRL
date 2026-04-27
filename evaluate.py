import numpy as np
from stable_baselines3 import PPO

from utils.dataset import VOCDataset
from detector.yolo_detector import YOLODetector
from utils.distortions import apply_distortion
from utils.metrics import load_gt_boxes, compute_detection_score, compute_f1, compute_avg_iou
from configs.config import *

import cv2


def evaluate(num_samples=100):

    print("🚀 Running evaluation...")

    dataset = VOCDataset("/home/ankit/datasets/VOC/images/train2007")
    detector = YOLODetector()
    model = PPO.load("models/object_rl_model")

    improvements = []
    successes = []
    recoveries = []

    f1_scores = []
    iou_scores = []

    alphas = []

    for i in range(num_samples):

        # -------- SAMPLE --------
        original, path = dataset.sample()
        original = cv2.resize(original, (IMG_SIZE, IMG_SIZE))

        # -------- DISTORT --------
        distorted = apply_distortion(original, 0.6, DISTORTION_TYPE)

        # -------- RL ACTION --------
        obs = distorted.astype("float32") / 255.0
        obs = obs.transpose(2, 0, 1)

        action, _ = model.predict(obs)
        alpha = float(action[0])
        alpha = max(0.3, min(2.0, alpha))

        alphas.append(alpha)

        transformed = apply_distortion(distorted, alpha, DISTORTION_TYPE)

        # -------- LOAD GT --------
        label_path = path.replace("images", "labels").replace(".jpg", ".txt")
        gt_boxes = load_gt_boxes(label_path, original.shape)

        # -------- DETECTIONS --------
        pred_o, _ = detector.predict(original)
        pred_d, _ = detector.predict(distorted)
        pred_s, _ = detector.predict(transformed)

        # -------- SCORES --------
        do = compute_detection_score(pred_o, gt_boxes, GAMMA)
        dd = compute_detection_score(pred_d, gt_boxes, GAMMA)
        ds = compute_detection_score(pred_s, gt_boxes, GAMMA)

        # -------- METRICS --------
        improvement = ds - dd
        success = 1 if ds > dd else 0
        recovery = 1 if (dd == 0 and ds > 0) else 0

        improvements.append(improvement)
        successes.append(success)
        recoveries.append(recovery)

        # extra metrics
        f1_scores.append(compute_f1(pred_s, gt_boxes))
        iou_scores.append(compute_avg_iou(pred_s, gt_boxes))

        print(
            f"[{i}] alpha={alpha:.2f} | dd={dd:.3f} → ds={ds:.3f} | Δ={improvement:.3f}"
        )

    # -------- FINAL STATS --------
    print("\n📊 FINAL RESULTS\n")

    print(f"Average Improvement (ds - dd): {np.mean(improvements):.4f}")
    print(f"Success Rate (ds > dd): {np.mean(successes)*100:.2f}%")
    print(f"Recovery Rate (dd=0 → ds>0): {np.mean(recoveries)*100:.2f}%")

    print("\n--- Detection Metrics ---")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
    print(f"Average IoU: {np.mean(iou_scores):.4f}")

    print("\n--- Alpha Stats ---")
    print(f"Mean Alpha: {np.mean(alphas):.3f}")
    print(f"Std Alpha: {np.std(alphas):.3f}")
    print(f"Min Alpha: {np.min(alphas):.3f}")
    print(f"Max Alpha: {np.max(alphas):.3f}")


if __name__ == "__main__":
    evaluate(num_samples=100)