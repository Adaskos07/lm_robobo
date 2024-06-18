from collections import deque
import numpy as np
import cv2

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete

from robobo_interface import SimulationRobobo

from data_files import FIGURES_DIR, RESULT_DIR, MODELS_DIR

def move_forward(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=speed, millis=duration)

def move_back(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=-speed, millis=duration)

def turn_left(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=speed, millis=duration)

def turn_right(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=-speed, millis=duration)


class SimEnv2(gym.Env):
    def __init__(self, rob: SimulationRobobo, max_steps, test_run=False):
        self.rob = rob
        self.test_run = test_run
        self.step_count = 0
        self.max_steps = max_steps

        self.action_space = Discrete(4)
        self.observation_space = MultiDiscrete([7,7,7,7,7,7,7,7])

    def step(self, action):
        action_map = {
            0: move_forward,
            1: move_back,
            2: turn_right,
            3: turn_left
        }
        action_map[action](self.rob, 60, 300)

        terminated = False
        truncated = False

        observation = self._get_obs()

        if self.is_collision(observation):
            reward = -200
            if not self.test_run:
                terminated = True
                self.rob.stop_simulation()
        elif self.step_count > self.max_steps:
            reward = 10
            truncated = True
            self.rob.stop_simulation()
        else:
            reward = 1

        self.step_count += 1
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.rob.set_phone_tilt(250, 100)
        self.rob.sleep(3)

        self.step_count = 0
        observation = self._get_obs()

        image = self.rob.get_image_front()
        # image = cv2.convertScaleAbs(image, alpha=(255.0))
        cv2.imwrite(MODELS_DIR / 'test_img.jpg', image)

        return observation, {}

    def close(self):
        self.rob.stop_simulation()
    
    def is_collision(self, irs):
        return int(np.mean(irs[2:5])) > 5 or int(np.mean(irs[[0,1,6]])) > 5
    
    def simplify_img(img):
        pass
    
    def _get_obs(self):
        irs = self.rob.read_irs()
        irs_discrete = np.digitize(irs, [0, 10, 50, 100, 200, 300])
        observation = irs_discrete
        return observation