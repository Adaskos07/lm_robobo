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
from .env2 import SimEnv2, HardEnv2, move_back, move_forward, turn_left, turn_right

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

                if terminated or truncated:
                # if truncated:
                    observation, info = env.reset()
        else:
            print(from_checkpoint)
            if from_checkpoint:
                model = DQN.load(MODELS_DIR / model_name, env=env)
            else:
                model = DQN("MultiInputPolicy", env, verbose=1,
                            learning_rate=0.01, gamma=0.6,
                            tensorboard_log=RESULT_DIR / f'{model_name}_train.log')
            model.learn(total_timesteps=2_000, log_interval=5)

            print('Saving model')
            model.save(MODELS_DIR / model_name)
            print('Model saved')
 
    if isinstance(rob, HardwareRobobo):
        print('Initialized hardware run')
        env = HardEnv2(rob, tilt_angle=100)
        model = DQN.load(MODELS_DIR / model_name)
        print('Starting actions')
        observation, info = env.reset()
        # print(observation)

        for _ in range(max_steps):
            action, _states = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action.item())