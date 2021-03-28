import os
import optuna
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
from optuna.visualization import plot_param_importances, plot_pareto_front
from optuna.study import StudyDirection
import numpy as np
import torch as th
from LCAEnv7 import LCAEnv, rundate
import matplotlib.pyplot as plt
import pandas as pd

from utils import plot_results

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
## todo: try optuna.samplers.NSGAIISampler
## Using Gaussian process because hyperparams are probably correlated
log_dir = "/tmp/DQN-LCAEnv/"
param_dir = "Param Tuning Results/"

os.makedirs(log_dir, exist_ok=True)


def objective(trial):

    """
    Create the object to pass to optuna to optimize
    """

    env = LCAEnv()
    env = Monitor(env, log_dir)

    lr_opt = trial.suggest_float("learning_rate", 0.00005,0.001)
    neurons_opt = trial.suggest_categorical("hidden_layer_neurons", [128, 256, 512])
    bs_opt = trial.suggest_categorical("batch_size", [1, 4, 16, 50, 128])
    tsteps_opt = trial.suggest_int("train_freq",1,50)
    gsteps_opt = trial.suggest_categorical("gradient_steps", [-1,1])




    # Now our typical stable baselines run
    policy_kwargs = dict(activation_fn=th.nn.GELU, net_arch=[neurons_opt, neurons_opt, neurons_opt])

    hyper_parameters = dict(learning_rate=lr_opt,
                            buffer_size=100000,
                            learning_starts=2000,
                            batch_size=bs_opt,
                            tau=1,  # 1 for hard update
                            gamma=0.99,
                            train_freq=(tsteps_opt, "step"),
                            gradient_steps=gsteps_opt,  # changed to -1 from 1
                            target_update_interval=20000,
                            exploration_initial_eps=0.3,
                            exploration_final_eps=0.001,
                            exploration_fraction=0.95,
                            max_grad_norm=0.8,
                            create_eval_env=False,
                            verbose=0,
                            seed=None, )

    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, **hyper_parameters)

    n_Steps = 800 #90,000
    model.learn(total_timesteps=n_Steps, log_interval=1000)


    # Validate the model
    reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=8) # 800 episodes

    # Return validation to optuna to select next env
    return reward_mean


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                storage= "sqlite:///HyperParamStudies.db",
                                study_name= f'DQN7 Study 1',
                                load_if_exists=True)

    study.optimize(objective, n_trials= 30)



# Name the script to run, and execute in parallel
from multiprocessing import Pool

processes = ('ParamTuningDistributed.py')


def run_process(process):
    os.system('python {}'.format(process))


pool = Pool(processes=3)
pool.map(run_process, processes)