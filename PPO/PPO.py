import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_results
from LCAEnv7 import LCAEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy

from typing import Callable

if __name__ == '__main__':


    def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """

        def _init() -> gym.Env:
            env = LCAEnv()
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init


    log_dir = "/tmp/PPO-LCAEnv/"
    os.makedirs(log_dir, exist_ok=True)


    env_id = "LCAEnv "
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])



    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=250000)
    model.save("PPO LCAEnv7")



    # Display results
    #results_plotter.plot_results([log_dir],1e10, results_plotter.X_EPISODES, "LCA Env 7")
    #plt.savefig("PPO  figure.jpg") #todo: f string
    #plt.show()

    #plot_results(log_dir) # Smoothed results
