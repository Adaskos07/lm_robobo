import cv2

from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from .env1 import SimEnv1

def main(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        env = SimEnv1(rob)
        model = DQN("CnnPolicy", env, verbose=1,
                    learning_rate=0.001)
        model.learn(total_timesteps=1000, log_interval=5)

        observation, info = env.reset()

        # for _ in range(100):
        #     # action = env.action_space.sample()
        #     action = model.predict(observation, deterministic=True)

        #     observation, reward, terminated, truncated, info = env.step(action)

        #     if terminated or truncated:
        #         observation, info = env.reset()

            # print(observation)

    # if isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)
