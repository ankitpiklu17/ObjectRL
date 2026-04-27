import numpy as np
import cv2
import random

def apply_distortion(img, alpha, mode):

    if mode == "brightness":
        return np.clip(alpha * img, 0, 255).astype(np.uint8)

    elif mode == "contrast":
        return np.clip(alpha * (img - 127) + 127, 0, 255).astype(np.uint8)

    elif mode == "blur":
        k = int(3 + alpha * 4)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(img, (k, k), 0)

    elif mode == "noise":
        noise = np.random.normal(0, alpha * 25, img.shape)
        noisy = img + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    return img


def random_strong_distortion(img):
    mode = random.choice(["brightness", "contrast", "blur", "noise"])

    # 🔥 strong distortion range
    alpha = random.uniform(0.2, 2.5)

    return apply_distortion(img, alpha, mode)