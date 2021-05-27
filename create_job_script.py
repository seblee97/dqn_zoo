import argparse
import time
import datetime
import os

from typing import List, Union


arg_parser = argparse.ArgumentParser()


arg_parser.add_argument(
    "--run_command", type=str, help="python command (or other) to run in cluster.", default=None
)
arg_parser.add_argument("--algorithm", type=str, help="name of algorithm to run", default="bootstrapped_dqn")
arg_parser.add_argument("--environment", type=str, help="name of atari env", default="pong")
arg_parser.add_argument("--penalty_type", type=str, help="name of penalty type", default="no_penalty")
arg_parser.add_argument("--save_path", type=str, help="path to save job script.")
arg_parser.add_argument("--num_cpus", type=int, help="number of CPUs to use for job.")
arg_parser.add_argument(
    "--conda_env_name", type=str, help="name of conda environment to load."
)
arg_parser.add_argument("--memory", type=int, help="memory per node.")
arg_parser.add_argument("--num_gpus", type=int, help="number of GPUs to use for job.")
arg_parser.add_argument("--gpu_type", type=str, help="type of GPU to use for job.")
# arg_parser.add_argument("--error_path", type=str, help="path of error file for job.")
# arg_parser.add_argument("--output_path", type=str, help="path of output file for job.")
arg_parser.add_argument("--modules", type=str, help="list of modules to load.", default=None)
arg_parser.add_argument("--num_hours", type=int, help="number of hours for runtime.", default=24)


arg_parser.add_argument(
    "--flat_chain", 
    type=int, 
    help="create a flat chain script with specified amount of repeats.", 
    default=None
)


def create_job_script(
    run_command: Union[None, str],
    algorithm: Union[None, str],
    environment: Union[None, str],
    penalty: Union[None, str],
    save_path: str,
    num_cpus: int,
    conda_env_name: str,
    memory: int,
    num_gpus: int,
    gpu_type: str,
    # error_path: str,
    # output_path: str,
    modules: List[str],
    results_folder: str,
    source_code_dir: str,
    walltime: str = "24:0:0",
) -> None:
    """Create a job script for use on HPC.

    Args:
            run_command: main script command, e.g. 'python run.py'
            save_path: path to save the job script to
            num_cpus: number of cores for job
            conda_env_name: name of conda environment to activate for job
            memory: number of gb memory to allocate to node.
            walltime: time to give job--1 day by default
    """
    with open(os.path.join(results_folder, save_path), "w") as file:
        resource_specification = f"#PBS -lselect=1:ncpus={num_cpus}:mem={memory}gb"
        if num_gpus:
            resource_specification += f":ngpus={num_gpus}:gpu_type={gpu_type}"
        file.write(f"{resource_specification}\n")
        file.write(f"#PBS -lwalltime={walltime}\n")
        # file.write("cd $PBS_O_WORKDIR\n")
        # output/error file paths
        file.write(f"#PBS -e {results_folder}/error.txt\n")
        file.write(f"#PBS -o {results_folder}/output.txt\n")
        # initialise conda env
        file.write("module load anaconda3/personal\n")
        # load other relevant modules, e.g. cuda
        for module in modules:
            file.write(f"module load {module}\n")        	
        file.write(f"source activate {conda_env_name}\n")
        # change to dir where job was submitted from
        # file.write("if [ -d results ]\n")
        # file.write("then\n")
        # file.write("    mkdir results\n")
        # file.write(f"if [ -d {results_folder} ]\n")
        # file.write("then\n")
        # file.write(f"    mkdir {results_folder}\n")
        # job script
        file.write(f"cd {source_code_dir}\n")
        run_command = run_command or (
            f"python -m dqn_zoo.{algorithm}.run_atari --environment_name {environment} "
            f"--shaping_function_type {penalty} "
            f"--results_csv_path {results_folder}/{algorithm}_{environment}_{penalty}.csv "
            f"--checkpoint_path {results_folder}/{algorithm}_{environment}_{penalty}.pkl"
            )
        file.write(f"{run_command}\n")


INT_STR_MAPPING = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def create_flat_chain_script(results_folder: str, script_name: str, num_repeats: int):
    full_script_name = os.path.join(results_folder, save_path)
    flat_chain_save_path = os.path.join(results_folder, f"flat_chain_{script_name}")
    with open(flat_chain_save_path, "w") as file:
        file.write("#!/bin/bash\n")
        file.write(f"{INT_STR_MAPPING[0]}=$(qsub {full_script_name})\n")
        file.write(f"echo ${INT_STR_MAPPING[0]}\n")
        for i in range(1, num_repeats):
            file.write(f"{INT_STR_MAPPING[i]}=$(qsub -W depend=afterok:${INT_STR_MAPPING[i-1]} {full_script_name})\n")
            file.write(f"echo ${INT_STR_MAPPING[i]}\n")


if __name__ == "__main__":
    args = arg_parser.parse_args()

    if args.modules is not None:
        modules = args.modules.split(",")
    else:
        modules = []

    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")

    cwd = os.getcwd()
    results_folder = os.path.join(cwd, "results", timestamp)

    os.makedirs(results_folder, exist_ok=True)argsz

    create_job_script(
        run_command=args.run_command,
        algorithm=args.algorithm,
        environment=args.environment,
        penalty=args.penalty_type,
        save_path=args.save_path,
        num_cpus=args.num_cpus,
        conda_env_name=args.conda_env_name,
        memory=args.memory,
        num_gpus=args.num_gpus,
        gpu_type=args.gpu_type,
        # error_path=args.error_path,
        # output_path=args.output_path,
        modules=modules,
        results_folder=results_folder,
        source_code_dir=cwd,
        walltime=f"{str(args.num_hours)}:0:0",
    )

    if args.flat_chain is not None:
        create_flat_chain_script(results_folder=results_folder, script_name=args.save_path, num_repeats=args.flat_chain)
