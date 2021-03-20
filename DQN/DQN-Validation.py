import numpy as np
import pandas as pd
from LCAEnv7 import LCAEnv, rundate
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


from stable_baselines3 import DQN

model_dir = "DQN Model Saves/"
results_dir = "DQN Results/"

model_name = "LCA Env 7 2021-03-19 09'54''"

env = LCAEnv()
model = DQN.load(model_dir + model_name)

# Reset and test in environment
episodes = 250
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
        action, states = model.predict(state, deterministic=True)
        state, reward, done, info = env.step(action)
        #print(action,state)
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


print("Average GWP Per Episode: {:.2f}\nAverage Cost Per Episode: {:.2f}".format(average_gwp,average_cost))


# Plot Results
fig, (ax1, ax2) = plt.subplots(2)

x = np.arange(episodes) + 1

ax1.plot(x, full_gwp_list, label='Average GWP Per Episode', color='red',marker='.',linestyle='none')
ax2.plot(x, full_cost_list, label='Average Cost Per Episode', color='blue',marker='.',linestyle='none')

# Averae lines
ax1.axhline(average_gwp,linestyle = 'dashed',color ='black')
ax2.axhline(average_cost,label = 'average', linestyle = 'dashed',color ='black')

# Set integer x-axis without displaying every tick necessarily
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

fig.suptitle("Testing Model Rewards")
fig.legend()
plt.savefig(results_dir + f"Testing Results {model_name}")
plt.show()


# Export results to CSV
np.set_printoptions(formatter={'float_kind':'{:.2f}'.format}) # state array from scientific to .4f

df_results = pd.DataFrame(full_info)
df_results.to_csv(results_dir + f"DQN Validation {model_name}.csv",float_format='%.2f')
