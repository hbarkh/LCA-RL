import os
import sys
import optuna
import numpy as np
import torch as th
from LCAEnv7 import LCAEnv
from stable_baselines3 import PPO
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
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 2048])
    bs_opt = trial.suggest_categorical("batch_size", [1, 16, 64]) # 1 best
    gamma = trial.suggest_categorical("gamma",[0.6,0.7,0.9,0.99])
    #gsteps_opt = trial.suggest_categorical("gradient_steps", [-1,1]) # -1 best




    # Now our typical stable baselines run
    policy_kwargs = dict(activation_fn=th.nn.GELU, net_arch=[512, 512, 512])

    hyper_parameters = dict(learning_rate=lr_opt,
                            n_steps=n_steps,
                            batch_size=bs_opt,
                            n_epochs=10,
                            gamma=gamma,
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
                            seed=None, )

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, **hyper_parameters)

    n_Steps = 40000
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
                                storage= "sqlite:///HyperParamStudiesPPO.db",
                                study_name= 'DQN7 Env Study PPO',
                                load_if_exists=True,
                                sampler=None) #TPE by default

    study.optimize(objective, n_trials= 3)
