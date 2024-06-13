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

from .task1 import test_sim
from .env1 import SimEnv1

def run_all_actions(rob: IRobobo):
    env = SimEnv1(rob)

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    if isinstance(rob, SimulationRobobo):
        print(rob.get_image_front().shape)
        print(rob.read_irs())
        test_sim(rob)


        observation, info = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset()

        print(observation)

    # if isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
