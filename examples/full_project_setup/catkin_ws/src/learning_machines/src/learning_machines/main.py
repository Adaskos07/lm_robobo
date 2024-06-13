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

def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    if isinstance(rob, SimulationRobobo):
        print(rob.get_image_front().shape)
        print(rob.read_irs())
        test_sim(rob)



    # if isinstance(rob, HardwareRobobo):
    #     test_hardware(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
