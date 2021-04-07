import os
import gym
import numpy as np
import torch as th
from LCAEnv7 import LCAEnv, rundate
import matplotlib.pyplot as plt

from utils import plot_results

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter

log_dir = "/tmp/DQN-LCAEnv/"
model_dir = "PPO Model Saves/"
results_dir = "PPO Results/"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_dir), exist_ok=True)
os.makedirs(os.path.dirname(results_dir), exist_ok=True)


# Make Environment
env = LCAEnv()
env = Monitor(env, log_dir)

# Set NN Architecture
#  actor (pi), value function (vf), Q function (qf)
#  ex. net_arch=[dict(pi=[32, 32], vf=[32, 32])])
#  to have networks share architecture use only net_arch
policy_kwargs = dict(activation_fn=th.nn.GELU, net_arch=[512,512,512])

# Set Hyper Parameters
hyper_parameters = dict(learning_rate=0.0003,
                        n_steps=2048,
                        batch_size=64,
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
                        tensorboard_log=None,
                        create_eval_env=False,
                        verbose=0,
                        seed=None,)





# Create model
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs,**hyper_parameters)

# Train Model
n_Steps = 1000
model.learn(total_timesteps=n_Steps, log_interval=1)

# Save Model
model.save(model_dir + f"LCA Env 7 {rundate}")

# Display results
results_plotter.plot_results([log_dir],1e10, results_plotter.X_EPISODES, "LCA Env 7")
plt.savefig(results_dir + f"LCA Env 7  - {rundate}.jpg")
plt.show()

plot_results(log_dir) # Smoothed results
