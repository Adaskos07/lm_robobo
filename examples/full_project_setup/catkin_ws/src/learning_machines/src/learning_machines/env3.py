import numpy as np

import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Dict

from robobo_interface import SimulationRobobo, IRobobo

from data_files import FIGURES_DIR, RESULT_DIR, MODELS_DIR

from .img_proc import get_dist
import cv2

def move_forward(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=speed, millis=duration)

def move_back(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=-speed, millis=duration)

def turn_left(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=speed, millis=duration)

def turn_right(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=-speed, millis=duration)


class SimEnv3(gym.Env):
    def __init__(self, rob: SimulationRobobo, max_steps, test_run=False):
        self.rob = rob
        self.step_count = 0
        self.max_steps = max_steps
        self.test_run = test_run

        self.total_reward = 0
        self.meal_count = 0

        self.img_width = 48

        self.action_space = Discrete(4)
        self.observation_space = Dict({
            'irs': MultiDiscrete([5,5,5,5,5,5,5,5]),
            # 'irs': MultiDiscrete([4,4,4,4,4,4,4,4]),
            'dist_r': Discrete(4),
            'dist_g': Discrete(4),
        })

        self.captured = False
        
    def step(self, action):
        print(self.rob.red_food_position(), self.rob.base_position(), self._calc_metric())
        action_map = {
            0: move_forward,
            1: move_back,
            2: turn_right,
            3: turn_left
        }
        # action_map[0](self.rob, 80, 500)
        if action < 2: # smaller step when rotating
            action_map[action](self.rob, 80, 500)
        else:
            action_map[action](self.rob, 30, 200)

        terminated = False
        truncated = False
        reward = -1

        observation = self._get_obs()

        # print(observation['irs'][4], observation['dist_r'])
        # if observation['irs'][4] > 2 and observation['dist_r'] < 2:
        #     self.captured = True

        # if self.captured:
        #     print('CAPTURED ', self.captured)

        if self.rob.base_detects_food():
            reward += 500
            terminated = True
            self.rob.stop_simulation()
        elif self.step_count > self.max_steps:
            reward += -self._calc_metric()
            truncated = True
            self.rob.stop_simulation()
        # else:
        #     if action == 1:
        #         self.captured = False
        #         reward += -10
        #     elif self.captured: 
        #         reward = -(observation['dist_g']) - 1
        #     else:
        #         reward = -(observation['dist_r']) - 2

        self.step_count += 1
        self.total_reward += reward
        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.rob.stop_simulation()
        self.rob.play_simulation()
        self.rob.sleep(0.2)
        self.total_reward = 0
        self.step_count = 0
        self.meal_count = 0
        observation = self._get_obs()
        return observation, {}

    def close(self):
        self.rob.stop_simulation()

    def _get_obs(self):
        irs = self.rob.read_irs()
        irs_discrete = self._digitize_irs(irs)

        img = self.rob.get_image_front()
        # if self.step_count == 1:
            # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(MODELS_DIR / 'red.jpg', hsv)
        distance_green = self._calc_dist(img)
        distance_red = self._calc_dist(img, green=False)

        observation = {'irs': irs_discrete, 'dist_g': distance_green, 'dist_r': distance_red}
        return observation
    
    def _digitize_irs(self, irs):
        # return np.digitize(irs, [10, 100, 250, 300])
        return np.digitize(irs, [250, 275, 285, 300])
        return np.digitize(irs, [100, 250, 300])
    
    def _calc_dist(self, img, green=True):
        distance = get_dist(img, size=self.img_width, green=green)
        # map distance to values from 0 to 3 where 3 means no object
        if distance is None:
            distance = 3
        else:
            distance = np.digitize(distance, [self.img_width // 5, self.img_width // 2.5])
        return distance
    
    def _calc_metric(self):
        base_xyz = self.rob.base_position()
        red_food_xyz = self.rob.red_food_position()

        xs = (base_xyz.x - red_food_xyz.x)**2
        ys = (base_xyz.y - red_food_xyz.y)**2
        distance = np.sqrt(xs + ys)
        # scale to usable numbers
        return int(distance * 100)
    

class HardEnv3():
    def __init__(self, rob: IRobobo, tilt_angle=100):
        self.rob = rob
        self.step_count = 0
        self.total_reward = 0
        self.meal_count = 0
        self.img_width = 48

        self.action_space = Discrete(4)
        self.observation_space = Dict({
            # 'irs': MultiDiscrete([5,5,5,5,5,5,5,5]),
            'irs': MultiDiscrete([4,4,4,4,4,4,4,4]),
            'dist_g': Discrete(4),
            'dist_r': Discrete(4)
        })

        # rob.set_phone_tilt(tilt_angle, 110)
        # rob.set_phone_pan(11, 100)
        # rob.set_phone_pan(110, 45)

    def reset(self):
        self.total_reward = 0
        self.step_count = 0
        self.meal_count = 0
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        action_map = {
            0: move_forward,
            1: move_back,
            2: turn_right,
            3: turn_left
        }
        if action < 2: # smaller step when rotating
            action_map[action](self.rob, 75, 500)
        else:
            action_map[action](self.rob, 20, 200)

        observation = self._get_obs()
        return observation, None, False, False, {}

    def _get_obs(self):
        irs = self.rob.read_irs()
        irs_discrete = self._digitize_irs(irs)

        img = self.rob.get_image_front()
        distance_green = self._calc_dist(img)
        distance_red = self._calc_dist(img, green=False)

        observation = {'irs': irs_discrete, 'dist_g': distance_green, 'dist_r': distance_red}
        return observation

    def _digitize_irs(self, irs):
        BackL, BackR, FrontL, FrontR, FrontC, FrontRR, BackC, FrontLL = self.rob.read_irs()

        # irs_BackL = np.digitize(BackL, [40, 68, 232, 2973])
        # irs_BackR = np.digitize(BackR, [27, 107, 297, 1002])
        # irs_FrontL = np.digitize(FrontL, [48, 93, 220, 934])
        # irs_FrontR = np.digitize(FrontR, [44, 89, 174, 1136])
        # irs_FrontC = np.digitize(FrontC, [21, 94, 192, 1868])
        # irs_FrontRR = np.digitize(FrontRR, [22, 60, 147, 944])
        # irs_BackC = np.digitize(BackC, [32, 101, 352, 1785])
        # irs_FrontLL = np.digitize(FrontLL, [19, 66, 220, 1063])


        irs_BackL = np.digitize(BackL, [68, 232, 2973])
        irs_BackR = np.digitize(BackR, [107, 297, 1002])
        irs_FrontL = np.digitize(FrontL, [93, 220, 934])
        irs_FrontR = np.digitize(FrontR, [89, 174, 1136])
        irs_FrontC = np.digitize(FrontC, [94, 192, 1868])
        irs_FrontRR = np.digitize(FrontRR, [60, 147, 944])
        irs_BackC = np.digitize(BackC, [101, 352, 1785])
        irs_FrontLL = np.digitize(FrontLL, [66, 220, 1063])

        irs_discrete = [irs_BackL, irs_BackR, irs_FrontL, irs_FrontR, irs_FrontC,
                        irs_FrontRR, irs_BackC, irs_FrontLL]
        return irs_discrete
    
    def _calc_dist(self, img, green=True):
        distance = get_dist(img, size=self.img_width, green=green)
        # map distance to values from 0 to 3 where 3 means no object
        if distance is None:
            distance = 3
        else:
            distance = np.digitize(distance, [self.img_width // 5, self.img_width // 2.5])
        return distance