import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Dict

from robobo_interface import SimulationRobobo

# from data_files import FIGURES_DIR, RESULT_DIR, MODELS_DIR

from .img_proc import get_dist

def move_forward(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=speed, millis=duration)

def move_back(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=-speed, millis=duration)

def turn_left(rob, speed, duration):
    rob.move_blocking(left_speed=-speed*0.4, right_speed=speed*0.4, millis=duration)

def turn_right(rob, speed, duration):
    rob.move_blocking(left_speed=speed*0.4, right_speed=-speed*0.4, millis=duration)


class SimEnv2(gym.Env):
    def __init__(self, rob: SimulationRobobo, max_steps, test_run=False):
        self.rob = rob
        self.test_run = test_run
        self.step_count = 0
        self.max_steps = max_steps

        self.total_reward = 0

        self.meal_count = 0
        self.img_width = 48

        self.action_space = Discrete(4)
        self.observation_space = Dict({
            'irs': MultiDiscrete([5,5,5,5,5,5,5,5]),
            'dist': Discrete(4)
        })
        
    def step(self, action):
        action_map = {
            0: move_forward,
            1: move_back,
            2: turn_right,
            3: turn_left
        }
        # smaller step when rotating
        action_map[action](self.rob, 75, 300)
        #else:
          #  action_map[action](self.rob, 30, 300)

        terminated = False
        truncated = False

        observation = self._get_obs()

        curr_meal_count = self.rob.nr_food_collected()
        if self.meal_count < curr_meal_count:
            # in case two or more meals were collected one after another
            nr_meals = curr_meal_count - self.meal_count
            reward = 50 * nr_meals

            self.meal_count = curr_meal_count

            if self.meal_count == 7:
                terminated = True
                reward += 25
                self.rob.stop_simulation()
        elif self.step_count > self.max_steps:
            reward = -25
            truncated = True
            self.rob.stop_simulation()
        elif action == 1:
            reward = -3
        else:
            reward = -(observation['dist'])

        self.step_count += 1
        self.total_reward += reward
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.total_reward = 0
        self.step_count = 0
        self.meal_count = 0

        self.rob.stop_simulation()
        self.rob.play_simulation()
        observation = self._get_obs()

        # image = self.rob.get_image_front()
        # # image = cv2.convertScaleAbs(image, alpha=(255.0))
        # cv2.imwrite(MODELS_DIR / 'test_img.jpg', image)

        return observation, {}

    def close(self):
        self.rob.stop_simulation()
    
    def is_food_consumed(self, irs, dist):
        '''Find out if there is a collision with green object
        '''
        return self.meal_count < self.rob.nr_food_collected()
        # for now we use simulations info

        # return any([i > 3 for i in irs])
    
    def _get_obs(self):
        irs = self.rob.read_irs()
        irs_discrete = np.digitize(irs, [10, 100, 200, 300])

        img = self.rob.get_image_front()
        distance = get_dist(img, size=self.img_width)
        # map distance to values from 0 to 3 where 3 means no object
        if distance is None:
            distance = 3
        else:
            distance = np.digitize(distance, [self.img_width // 4, self.img_width // 2])

        observation = {
            'irs': irs_discrete,
            'dist': distance
        }
        return observation