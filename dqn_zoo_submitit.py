import subprocess

import submitit

executor = submitit.AutoExecutor(folder="log_exp")

executor.update_parameters(timeout_min=120, mem_gb=30, gpus_per_node=1, cpus_per_task=8, slurm_array_parallelism=256, slurm_partition="gpu")

def run_expt(algorithm: str, environment_group: str, environment: str = None, csv_path: str = None, checkpoint_path: str = None):
    command = f"python -m dqn_zoo.{algorithm}.run_{environment_group}"
    if environment is not None:
        command = f"{command} --environment_name {environment}"
    if csv_path is not None:
        command = f"{command} --results_csv_path {csv_path}"
    if checkpoint_path is not None:
        command = f"{command} --checkpoint_path {checkpoint_path}"
    subprocess.call(command)

jobs = []
with executor.batch():
    for seed in range(1):
        job = executor.submit(run_expt, algorithm="dqn", environment_group="atari", environment="pong", csv_path="results/dqn_trial_new.csv", checkpoint_path="results/dqn_trial_new.pkl")
