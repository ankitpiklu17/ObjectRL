# Object RL for Detection Improvement


This work implements a **Reinforcement Learning (RL) framework** that learns image transformations to improve object detection performance under challenging conditions.

Instead of modifying the detector, we **optimize the input image** so that a fixed detector performs better.

---

## Objective

Improve detection performance under distortions by maximizing:

```
d(x) = γ · IoU + (1 − γ) · F1
```

Reward function:

```
β = 2d_s − d_o − d_d
```

Where:

* `d_o` = score on original image
* `d_d` = score on distorted image
* `d_s` = score after RL transformation

---

## 🏗️ Project Structure

```
object_rl/
│
├── train.py              # RL training script
├── evaluate.py           # Evaluation metrics
├── inference.py          # Run trained model
│
├── configs/              # Configuration files
├── env/                  # RL environment
├── utils/                # Dataset, metrics, distortions
├── detector/             # YOLO detector wrapper
```



##  Dataset

This project uses the **PASCAL VOC dataset**.


##  Training

```
python train.py
```

The RL agent learns to select transformation parameters that improve detection.

---


