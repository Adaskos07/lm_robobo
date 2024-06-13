from gymnasium import gym
from gymnsium import spaces

from robobo_interface import SimulationRobobo

class SimEnv1(gym.Env):
    def __init__(self, rob: SimulationRobobo):
        self.rob = rob
        self.action_spaces = spaces.Discrete(4)

    def step(self, action):
        # termination condition check



        # reward

        # 
        return observation, reward, terminated, False, info

    
    def reset(self, seed=None):
        super().reset(seed=seed)



        observation = self._get_obs()
        return observation
    
    def _get_obs(self):
        # get info
        position = self.rob.get_position() # not available irl
        ir_reads = self.rob.read_irs()
        image = self.get_image_front()
        time = self.rob.get_sim_time()

        # convert info to observation space
        observation = {}
        return observation, None