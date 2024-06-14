import cv2

from data_files import FIGRURES_DIR, RESULT_DIR
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

from .env1 import SimEnv1, move_back, move_forward, turn_left, turn_right

def robot_run(rob: IRobobo, max_steps, test_run=False):
    if isinstance(rob, SimulationRobobo):
        env = SimEnv1(rob, max_steps=max_steps, test_run=test_run)
        # check_env(env, warn=True)

        if test_run:
            model = DQN.load(RESULT_DIR / 'dqn_model')
            observation, info = env.reset()
            while True:
                # action = env.action_space.sample()
                action, _states = model.predict(observation, deterministic=True)

                observation, reward, terminated, truncated, info = env.step(action.item())

                # if terminated or truncated:
                if truncated:
                    observation, info = env.reset()
        else:
            model = DQN("MlpPolicy", env, verbose=1,
                        learning_rate=0.01)
            model.learn(total_timesteps=1000, log_interval=5)

            print('Saving model')
            model.save(RESULT_DIR / 'dqn_model')
            print('Model saved')
 
    if isinstance(rob, HardwareRobobo):
        model = DQN.load(RESULT_DIR / 'dqn_model')

        for _ in range(max_steps):
            action, _states = model.predict(observation, deterministic=True)
            action = action.item()
            action_map = {
                0: move_forward,
                1: move_back,
                2: turn_right,
                3: turn_left
            }
            action_map[action](rob, 60, 300)