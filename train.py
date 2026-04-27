import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from env.object_rl_env import ObjectRLEnv
from utils.dataset import VOCDataset
from detector.yolo_detector import YOLODetector
from configs.config import TOTAL_TIMESTEPS


def make_env():
    dataset = VOCDataset("/home/ankit/datasets/VOC/images/train2007")
    detector = YOLODetector()
    return ObjectRLEnv(dataset, detector)


def main():
    print("🚀 Training with HARD distortions...")

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        ent_coef=0.01,
        device="cuda"
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    os.makedirs("models", exist_ok=True)
    model.save("models/object_rl_model")

    print("✅ Training complete.")


if __name__ == "__main__":
    main()