import os
import gym
from LCAEnv7 import LCAEnv
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

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
    num_cpu = 6  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=[dict(pi=[64, 64, 64], vf=[128, 128, 128])])

    kwargs = dict( learning_rate=0.0003,
                   n_steps=50,
                   batch_size=2,
                   n_epochs=10,
                   gamma=0.99,
                   gae_lambda=0.95,
                   clip_range=0.2,
                   clip_range_vf=None,
                   ent_coef=0.0,
                   vf_coef=0.5,
                   max_grad_norm=0.5,
                   use_sde=False,
                   sde_sample_freq=- 1,
                   target_kl=None,
                   create_eval_env=False,
                   policy_kwargs=policy_kwargs,
                   seed=None, )


    model = PPO("MlpPolicy", env, verbose=1 , **kwargs)

    model.learn(total_timesteps=100000,eval_log_path = log_dir)
    model.save("PPO LCAEnv7")



    # Display results
    #results_plotter.plot_results([log_dir],1e10, results_plotter.X_EPISODES, "LCA Env 7")
    #plt.savefig("PPO  figure.jpg") #todo: f string
    #plt.show()

    #plot_results(log_dir) # Smoothed results
