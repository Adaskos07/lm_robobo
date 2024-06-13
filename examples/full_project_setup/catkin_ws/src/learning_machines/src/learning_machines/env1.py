import numpy as np

from gymnasium import gym
from gymnsium.spaces import Box, Tuple, Dict, Discrete, MultiDiscrete

from robobo_interface import SimulationRobobo, Position, Orientation

from .task1 import move_forward, move_back, move_left, move_right


class SimEnv1(gym.Env):
    def __init__(self, rob: SimulationRobobo):
        self.rob = rob
        self.action_space = Discrete(4)
        self.observation_space = Dict({
            'image': Box(low=0, high=0, shape=(512, 512, 3), dtype=np.uint8),

            'irs': MultiDiscrete([6,6,6,6,6,6,6,6]),

            'time': Box(high=50_000, dtype=float),

            # 'position': Tuple(
            #     Box(dtype=float),
            #     Box(dtype=float),
            #     Box(dtype=float),
            # )
        })

    def step(self, action):
        action_map = {
            0: move_forward,
            1: move_back,
            2: move_right,
            3: move_left
        }

        action_map[action](self.rob, 1000, 1000)

        # termination condition check
        terminated = False
        truncated = False

        observation = self._get_obs()
        reward = -1

        return observation, reward, terminated, truncated, None

    
    def reset(self, seed=None):
        super().reset(seed=seed)
        start_position = Position(x=0.0, y=0.0, z=0.09)  # Set the starting position
        start_orientation = Orientation(yaw=-175.00036138789557,
                                        pitch=-19.996487020842473,
                                        roll=4.820286812070959e-05)  
        self.rob.set_position(start_position, start_orientation)

        observation = self._get_obs()
        return observation
    
    def _get_obs(self):
        position = self.rob.get_position() # not available irl
        irs = self.rob.read_irs()
        image = self.get_image_front()
        time = self.rob.get_sim_time()

        irs_discrete = np.digitize(irs, [0.0, 2000, 4000, 6000, 8000, 10_000])

        # convert info to observation space
        observation = {
            'image': image,
            'irs': irs_discrete,
            'time': time
        }
        return observation, None
    