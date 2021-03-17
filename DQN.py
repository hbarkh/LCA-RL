import os
import gym
import torch as th
from LCAEnv7 import LCAEnv
import matplotlib.pyplot as plt

from utils import plot_results

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)


# Make Environment
env = LCAEnv()
env = Monitor(env, log_dir)

# Set NN Architecture
#  actor (pi), value function (vf), Q function (qf)
#  ex. net_arch=[dict(pi=[32, 32], vf=[32, 32])])
#  to have networks share architecture use only net_arch
policy_kwargs = dict(activation_fn=th.nn.GELU, net_arch=[256, 256, 256])

# Set Hyper Parameters
hyper_parameters = dict(learning_rate=0.001,
                        buffer_size=1000000,
                        learning_starts=1000,
                        batch_size=32,
                        tau=1.0, #1 for hard update
                        gamma=0.99,
                        train_freq=4,
                        gradient_steps=1,
                        target_update_interval=10,
                        exploration_initial_eps=1.0,
                        exploration_final_eps=0.005,
                        exploration_fraction=1,
                        max_grad_norm=10,
                        tensorboard_log=None,
                        create_eval_env=False,
                        verbose=1,
                        seed=None,)





# Create model
model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs,**hyper_parameters)

# Train Model
model.learn(total_timesteps=100000, log_interval=4)

# Save Model
model.save("LCA Env 7")

# Display results
results_plotter.plot_results([log_dir],1e10, results_plotter.X_EPISODES, "LCA Env 7")
plt.show()

plot_results(log_dir) # Smoothed results

del model # remove to demonstrate saving and loading

model = DQN.load("LCA Env 7")

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)


    #env.render()
    if done:
      obs = env.reset()
