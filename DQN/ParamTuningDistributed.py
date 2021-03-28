import os
import sys
import optuna
import torch as th
from LCAEnv7 import LCAEnv
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

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

    n_Steps = 90000
    model.learn(total_timesteps=n_Steps, log_interval=None)


    # Validate the model
    reward_mean, reward_std = evaluate_policy(model, env, n_eval_episodes=1000)

    # Return validation to optuna to select next env
    return reward_mean


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize",
                                storage= "sqlite:///HyperParamStudies.db",
                                study_name= f'DQN7 Study 1',
                                load_if_exists=True)

    study.optimize(objective, n_trials= 10)
