import numpy as np
import pandas as pd
from LCAEnv7 import LCAEnv, rundate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


env = LCAEnv()


# Reset and test in environment
episodes = 10
full_gwp_list = []
full_cost_list = []
full_info = []
eps = 0.2

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    episode_gwp_list = []
    episode_cost_list = []

    while not done:
        # env.render()
        e = np.random.rand()
        if e < eps:
            action = env.action_space.sample()
        else:
            action = 6
        state, reward, done, info = env.step(action)
        print(action)
        print(state)
        episode_gwp_list.append(reward)
        episode_cost_list.append(info['total_cost'])
        full_info.append(info)

    gwp = -np.average(episode_gwp_list)
    cost = np.average(episode_cost_list)
    #print('Episode:{} Score:{:.2f} Cost: {:.2f}'.format(episode, gwp, cost))

    full_gwp_list.append(gwp)
    full_cost_list.append(cost)



average_gwp = np.average(full_gwp_list)
average_cost = np.average(full_cost_list)