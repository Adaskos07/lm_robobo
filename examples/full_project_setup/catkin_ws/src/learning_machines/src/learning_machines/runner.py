import cv2

from data_files import FIGURES_DIR, RESULT_DIR, MODELS_DIR
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

# from .env1 import SimEnv1, move_back, move_forward, turn_left, turn_right
from .env2 import SimEnv2, move_back, move_forward, turn_left, turn_right

def robot_run(rob: IRobobo, max_steps,
              test_run=False, model_name=None, from_checkpoint=False):
    if isinstance(rob, SimulationRobobo):
        # env = SimEnv1(rob, max_steps=max_steps, test_run=test_run)
        env = SimEnv2(rob, max_steps=max_steps, test_run=test_run)
        # check_env(env, warn=True)
        if test_run:
            model = DQN.load(MODELS_DIR / model_name)
            observation, info = env.reset()
            while True:
                action, _states = model.predict(observation, deterministic=True)

                observation, reward, terminated, truncated, info = env.step(action.item())

                # if terminated or truncated:
                if truncated:
                    observation, info = env.reset()
        else:
            if from_checkpoint:
                model = DQN.load(MODELS_DIR / model_name, env=env)
            else:
                model = DQN("MlpPolicy", env, verbose=1,
                            learning_rate=0.01)
            model.learn(total_timesteps=50, log_interval=5)

            print('Saving model')
            model.save(MODELS_DIR / model_name)
            print('Model saved')
 
    if isinstance(rob, HardwareRobobo):
        model = DQN.load(MODELS_DIR / model_name)

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