import cv2
from stable_baselines3 import PPO

from utils.dataset import VOCDataset
from detector.yolo_detector import YOLODetector
from utils.distortions import apply_distortion
from configs.config import DISTORTION_TYPE


def draw_boxes(img, boxes):
    out = img.copy()
    for b in boxes:
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(out, (x1,y1),(x2,y2),(0,255,0),2)
    return out


def main():
    model = PPO.load("object_rl_model")

    dataset = VOCDataset("/home/ankit/datasets/VOC/images/train2007")
    detector = YOLODetector()

    original, _ = dataset.sample()

    distorted = apply_distortion(original, 0.6, DISTORTION_TYPE)

    obs = distorted.astype("float32") / 255.0
    obs = obs.transpose(2,0,1)

    action, _ = model.predict(obs)
    alpha = max(0.3, float(action[0]))

    transformed = apply_distortion(distorted, alpha, DISTORTION_TYPE)

    boxes, _ = detector.predict(transformed)

    result = draw_boxes(transformed, boxes)

    cv2.imwrite("result.jpg", result)

    print(f"Predicted alpha: {alpha:.3f}")
    print("Saved result.jpg")


if __name__ == "__main__":
    main()