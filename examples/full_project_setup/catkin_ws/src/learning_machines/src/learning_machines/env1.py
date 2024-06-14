import numpy as np
import cv2

import gymnasium as gym
from gymnasium.spaces import Box, Tuple, Dict, Discrete, MultiDiscrete

from robobo_interface import SimulationRobobo, Position, Orientation


def move_forward(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=speed, millis=duration)

def move_back(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=-speed, millis=duration)

def turn_left(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=speed, millis=duration)

def turn_right(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=-speed, millis=duration)


class SimEnv1(gym.Env):
    def __init__(self, rob: SimulationRobobo):
        self.rob = rob
        self.step_count = 0
        self.action_space = Discrete(4)
        self.max_steps = 20
        # self.observation_space = Box(low=0, high=0, shape=(512, 512, 3), dtype=np.uint8),
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)


        # self.observation_space = Dict({
        #     'image': Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8),

        #     'irs': MultiDiscrete([6,6,6,6,6,6,6,6]),

        #     'time': Box(low=0, high=50_000, dtype=float),

        #     # 'position': Tuple(
        #     #     Box(dtype=float),
        #     #     Box(dtype=float),
        #     #     Box(dtype=float),
        #     # )
        # })

    def step(self, action):
        action_map = {
            0: move_forward,
            1: move_back,
            2: turn_right,
            3: turn_left
        }

        action_map[action](self.rob, 100, 500)
        # self.rob.move_blocking()
        # termination condition check
        terminated = False
        truncated = False
        reward = -1

        if self.step_count > self.max_steps:
            truncated = True
            self.rob.stop_simulation()
        elif self.rob.read_irs()[4] > 500 or self.rob.read_irs()[3] > 500:
            terminated = True
            reward = 100
            self.rob.stop_simulation()


        observation = self._get_obs()
        self.step_count += 1
        return observation, reward, terminated, truncated, {}

    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rob.stop_simulation()
        self.rob.play_simulation()


        start_position = Position(x=0.0, y=0.0, z=0.09)  # Set the starting position
        start_orientation = Orientation(yaw=-175.00036138789557,
                                        pitch=-19.996487020842473,
                                        roll=4.820286812070959e-05)  
        self.rob.set_position(start_position, start_orientation)
        self.step_count = 0

        observation = self._get_obs()
        return observation, {}
    
    def _get_obs(self):
        position = self.rob.get_position() # not available irl
        # irs = self.rob.read_irs()
        image = self.rob.get_image_front()
        # image = cv2.resize(image, (32, 32))[:,:,2]
        image = cv2.resize(image, (64, 64))
    
        time = self.rob.get_sim_time()

        # irs_discrete = np.digitize(irs, [0.0, 2000, 4000, 6000, 8000, 10_000])

        # convert info to observation space
        # observation = {
        #     'image': image,
        #     'irs': irs_discrete,
        #     'time': time
        # }
        observation = image
        return observation

    