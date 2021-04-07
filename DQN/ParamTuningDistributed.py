import os
import sys
import optuna
import numpy as np
import torch as th
from LCAEnv7 import LCAEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from optuna.samplers import NSGAIISampler

"""
This version of param tuning simply removes plots and disables printing
It will be called from another python file to run in parallel
Optuna when connected to a database and pointed to study
recognizes the past trials, does its magic, learns from past trials,
and writes to new trials. Voila
"""
# Disable print statement
sys.stdout = open(os.devnull, 'w')


param_dir = "Param Tuning Results/"



def objective(trial):

    """
    Create the object to pass to optuna to optimize
    """

    env = LCAEnv()
    env = Monitor(env)

    lr_opt = trial.suggest_float("learning_rate", 0.00001,0.001)
    bs_opt = trial.suggest_categorical("batch_size", [1, 4, 16])
    tsteps_opt = trial.suggest_int("train_freq",1,20)
    gamma_opt = trial.suggest_categorical("gamma", [0.6, 0.8, 0.9, 0.99])
    target_opt = trial.suggest_categorical("target", [10,100,1000,10000,20000])
    max_grad_norm = trial.suggest_categorical("max_grad", [0.1,0.4,0.6,0.8,1,1.5,2.5,5])




    # Now our typical stable baselines run
    policy_kwargs = dict(activation_fn=th.nn.GELU, net_arch=[512, 512, 512])

    hyper_parameters = dict(learning_rate=lr_opt,
                            buffer_size=100000,
                            learning_starts=2000,
                            batch_size=bs_opt,
                            tau=1,  # 1 for hard update
                            gamma=gamma_opt,
                            train_freq=(tsteps_opt, "step"),
                            gradient_steps=-1,
                            target_update_interval=target_opt,
                            exploration_initial_eps=0.3,
                            exploration_final_eps=0.001,
                            exploration_fraction=0.95,
                            max_grad_norm=max_grad_norm,
                            create_eval_env=False,
                            verbose=0,
                            seed=None, )

    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, **hyper_parameters)

    n_Steps = 50000
    model.learn(total_timesteps=n_Steps, log_interval=None)


    # Validate the model
    """
    Validating step-wise instead of using model.evaluate due to
    difference in outcomes. To be investigated later.
    """

    episodes = 1000
    full_gwp_list = []
    full_cost_list = []

    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        episode_gwp_list = []
        episode_cost_list = []

        while not done:

            action, states = model.predict(state, deterministic=True)
            state, reward, done, info = env.step(action)

            episode_gwp_list.append(info['total_gwp'])
            episode_cost_list.append(info['total_cost'])


        gwp = -np.average(episode_gwp_list)
        cost = np.average(episode_cost_list)


        full_gwp_list.append(gwp)
        full_cost_list.append(cost)

    average_gwp = np.average(full_gwp_list)
    average_cost = np.average(full_cost_list)



    # Alternative method is using 1 liner evaluate_policy below:
    #reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=1000)

    # Return validation to optuna to select next env
    return average_gwp


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                storage= "sqlite:///HyperParamStudies.db",
                                study_name= 'DQN7 Study 3',
                                load_if_exists=True,
                                sampler=None) #TPE by default

    study.optimize(objective, n_trials= 10)
