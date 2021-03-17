import os
import gym
from utils import plot_results
import matplotlib.pyplot as plt

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.evaluation import evaluate_policy

log_dir = "/tmp/gym/"
tb_dir = "./tensorboard/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(tb_dir, exist_ok=True) # tf does not work for macs :(

env = gym.make("CartPole-v0")
env = Monitor(env, log_dir)

model = DQN("MlpPolicy", env, verbose=0,tensorboard_log=tb_dir) # no tb on mac due to tensorflow dependancy
model.learn(total_timesteps=1500, log_interval=4)
model.save("dqn_pendulum")

results_plotter.plot_results([log_dir], 1e5, results_plotter.X_TIMESTEPS, "Cartpole")
plt.show()

plot_results(log_dir)

del model # remove to demonstrate saving and loading

model = DQN.load("dqn_pendulum")

obs = env.reset()
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)


    #env.render()
    if done:
      obs = env.reset()