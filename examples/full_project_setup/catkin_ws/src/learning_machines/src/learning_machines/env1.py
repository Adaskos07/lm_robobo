import numpy as np

from gymnasium import gym
from gymnsium import spaces

from robobo_interface import SimulationRobobo, Position, Orientation


class SimEnv1(gym.Env):
    def __init__(self, rob: SimulationRobobo):
        self.rob = rob
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=0, shape=(512, 512, 3), dtype=np.uint8),
            'irs': spaces.Tuple(
                spaces.Box(high=10000.0, dtype=float),
                spaces.Box(high=10000.0, dtype=float),
                spaces.Box(high=10000.0, dtype=float),
                spaces.Box(high=10000.0, dtype=float),
                spaces.Box(high=10000.0, dtype=float),
                spaces.Box(high=10000.0, dtype=float),
                spaces.Box(high=10000.0, dtype=float),
                spaces.Box(high=10000.0, dtype=float),
            ),
            'time': spaces.Box(high=50_000, dtype=float)
        })

    def step(self, action):
        # termination condition check



        # reward

        # 
        # return observation, reward, terminated, False, info
        pass

    
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
        # get info
        position = self.rob.get_position() # not available irl
        ir_reads = self.rob.read_irs()
        image = self.get_image_front()
        time = self.rob.get_sim_time()

        # convert info to observation space
        observation = {
            'image': image,
            'irs': tuple(ir_reads),
            'time': time
        }
        return observation, None