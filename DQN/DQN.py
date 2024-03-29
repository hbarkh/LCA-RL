import os
import gym
import numpy as np
import torch as th
from LCAEnv7 import LCAEnv, rundate
import matplotlib.pyplot as plt

from utils import plot_results

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter

log_dir = "/tmp/DQN-LCAEnv/"
model_dir = "DQN Model Saves/"
results_dir = "DQN Results/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
os.makedirs(os.path.dirname(results_dir), exist_ok=True)

gwp_weights = [0,10,20,30,40,80,90,100]


for gwp_weight in gwp_weights:

    # fix for file name
    gwp_weight = int(gwp_weight)
    print(gwp_weight)

    # Make Environment
    env = LCAEnv(gwp_weight=gwp_weight/100)
    env = Monitor(env, log_dir)

    # Set NN Architecture
    #  actor (pi), value function (vf), Q function (qf)
    #  ex. net_arch=[dict(pi=[32, 32], vf=[32, 32])])
    #  to have networks share architecture use only net_arch
    policy_kwargs = dict(activation_fn=th.nn.GELU, net_arch=[512,512,512])

    # Set Hyper Parameters
    hyper_parameters = dict(learning_rate=0.00086,
                            buffer_size=50000,
                            learning_starts=2000,
                            batch_size=16,
                            tau=1, #1 for hard update
                            gamma=0.9,
                            train_freq=(20, "step"),
                            gradient_steps=-1, #changed to -1 from 1
                            target_update_interval=10,
                            exploration_initial_eps=0.3,
                            exploration_final_eps=0.001,
                            exploration_fraction=0.95,
                            max_grad_norm=0.8,
                            create_eval_env=False,
                            verbose=0,
                            seed=None,)





    # Create model
    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs,**hyper_parameters)

    # Train Model
    n_Steps = 100000
    model.learn(total_timesteps=n_Steps, log_interval=1)

    # Save Model
    model.save(model_dir + f"LCA Env 7 GWP Weight {gwp_weight}")

    # Display results
    #results_plotter.plot_results([log_dir],1e10, results_plotter.X_EPISODES, "LCA Env 7")
    #plt.savefig(results_dir + f"LCA Env 7  - {rundate}.jpg")
    #plt.show()

    plot_results(log_dir,title =f"GWP Weight {gwp_weight}") # Smoothed results
