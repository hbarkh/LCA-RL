import numpy as np
from LCAEnv7 import LCAEnv
import matplotlib.pyplot as plt


from stable_baselines3 import DQN


env = LCAEnv()
model = DQN.load("LCA Env 7")

# Reset and test in environment
episodes = 20
full_gwp_list = []
full_cost_list = []
full_info = []

for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    episode_gwp_list = []
    episode_cost_list = []

    while not done:
        # env.render()
        action, _states = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        # print(action,n_state)
        episode_gwp_list.append(reward)
        episode_cost_list.append(info['total_cost'])
        full_info.append(info)

    gwp = -np.average(episode_gwp_list)
    cost = np.average(episode_cost_list)
    print('Episode:{} Score:{:.2f} Cost: {:.2f}'.format(episode, gwp, cost))

    full_gwp_list.append(gwp)
    full_cost_list.append(cost)

print("Average GWP Per Episode: {:.2f}\nAverage Cost Per Episode: {:.2f}".format(np.average(full_gwp_list),np.average(full_cost_list)))

plt.plot(full_gwp_list, label='Average GWP Per Episode', color='red',marker='.',linestyle='none')
plt.plot(full_cost_list, label='Average Cost Per Episode', color='blue',marker='.',linestyle='none')

plt.legend()
plt.title("Testing Model Rewards")
plt.show()