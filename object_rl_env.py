import gymnasium as gym
from gymnasium.spaces import Discrete, Box
import numpy as np
import cv2
import random

from utils.distortions import apply_distortion, random_strong_distortion
from utils.metrics import load_gt_boxes, compute_detection_score
from configs.config import *


class ObjectRLEnv(gym.Env):

    def __init__(self, dataset, detector):
        super().__init__()

        self.dataset = dataset
        self.detector = detector

        self.action_space = Discrete(len(ALPHA_VALUES))

        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(IMG_SIZE * IMG_SIZE * 3,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 🔥 keep sampling until HARD case found
        while True:
            self.original, self.img_path = self.dataset.sample()
            self.original = cv2.resize(self.original, (IMG_SIZE, IMG_SIZE))

            # 🔥 strong distortion
            self.distorted = random_strong_distortion(self.original)

            dd = self.evaluate(self.distorted)

            # 🔥 keep only hard samples
            if dd < 0.5:
                break

        obs = self.distorted.astype("float32") / 255.0
        obs = obs.flatten()

        return obs, {}

    def evaluate(self, img):
        pred_boxes, _ = self.detector.predict(img)

        label_path = self.img_path.replace("images", "labels").replace(".jpg", ".txt")
        gt_boxes = load_gt_boxes(label_path, img.shape)

        return compute_detection_score(pred_boxes, gt_boxes, GAMMA)

    def step(self, action):
        alpha = ALPHA_VALUES[action]

        transformed = apply_distortion(self.distorted, alpha, "brightness")

        do = self.evaluate(self.original)
        dd = self.evaluate(self.distorted)
        ds = self.evaluate(transformed)

        beta = 2 * ds - do - dd

        reward = beta / 2.0
        reward = np.clip(reward, -1.0, 1.0)

        done = True

        obs = transformed.astype("float32") / 255.0
        obs = obs.flatten()

        print(
            f"alpha={alpha:.2f} | dd={dd:.3f} → ds={ds:.3f} | reward={reward:.3f}"
        )

        return obs, reward, done, False, {}