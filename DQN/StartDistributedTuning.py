from multiprocessing import Pool
import os

# Run Multi Processes
num_processes = 2
processes = ('ParamTuningDistributed.py',) * num_processes


def run_process(process):
    os.system(f'python {process}')


pool = Pool(processes=num_processes)
pool.map(run_process, processes)


# Once complete, make the plots and save
import optuna
import datetime
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
from optuna.visualization import plot_param_importances

rundate = str(datetime.datetime.now().strftime("%Y-%m-%d %H'%M''"))

param_dir = "Param Tuning Results/"
os.makedirs(param_dir, exist_ok=True)

study = optuna.study.load_study(study_name='DQN7 Study 3', storage="sqlite:///HyperParamStudies.db")

fig = plot_parallel_coordinate(study)
fig.write_html(param_dir + f'parallel coordinates {rundate}.html' )
fig.show()

fig = plot_optimization_history(study)
fig.write_html(param_dir + f'optimization history {rundate}.html')
fig.show()

fig = plot_param_importances(study)
fig.write_html(param_dir + f'Param importances  {rundate}.html') # for bi or tri-objective studies
fig.show()
