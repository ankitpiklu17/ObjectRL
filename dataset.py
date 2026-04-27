import os
import cv2
import random

class VOCDataset:
    def __init__(self, root):
        self.img_paths = []

        for root_dir, _, files in os.walk(root):
            for f in files:
                if f.endswith(".jpg"):
                    self.img_paths.append(os.path.join(root_dir, f))

    def sample(self):
        path = random.choice(self.img_paths)
        img = cv2.imread(path)
        img = cv2.resize(img, (128, 128))
        return img, path