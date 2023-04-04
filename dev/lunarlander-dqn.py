
import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
     

gym_name = "LunarLander-v2"
env = gym.make(gym_name)

tensorboard_log = "data/tb/"

dqn_model = DQN("MlpPolicy",
            env,
            verbose=1,
            train_freq=16,
            gradient_steps=8,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=600,
            learning_starts=1000,
            buffer_size=100000,
            batch_size=128,
            learning_rate=0.0001,
            policy_kwargs=dict(net_arch=[64, 64]),
            tensorboard_log=tensorboard_log,
            seed=2)

mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), deterministic=True, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

dqn_model.learn(int(1.2e5), log_interval=10)
mean_reward, std_reward = evaluate_policy(dqn_model, dqn_model.get_env(), deterministic=True, n_eval_episodes=20)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")