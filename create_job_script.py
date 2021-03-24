import argparse


arg_parser = argparse.ArgumentParser()


arg_parser.add_argument(
    "--run_command", type=str, help="python command (or other) to run in cluster."
)
arg_parser.add_argument("--save_path", type=str, help="path to save job script.")
arg_parser.add_argument("--num_cpus", type=int, help="number of CPUs to use for job.")
arg_parser.add_argument(
    "--conda_env_name", type=str, help="name of conda environment to load."
)
arg_parser.add_argument("--memory", type=int, help="memory per node.")
arg_parser.add_argument("--num_gpus", type=int, help="number of GPUs to use for job.")
arg_parser.add_argument("--gpu_type", type=str, help="type of GPU to use for job.")
arg_parser.add_argument("--error_path", type=str, help="path of error file for job.")
arg_parser.add_argument("--output_path", type=str, help="path of output file for job.")
arg_parser.add_argument("--gpu_type", type=str, help="type of GPU to use for job.")


def create_job_script(
    run_command: str,
    save_path: str,
    num_cpus: int,
    conda_env_name: str,
    memory: int,
    num_gpus: int,
    gpu_type: str,
    error_path: str,
    output_path: str,
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
    with open(save_path, "w") as file:
        resource_specification = f"#PBS -lselect=1:ncpus={num_cpus}:mem={memory}gb"
        if num_gpus:
            resource_specification += f":ngpus={num_gpus}:gpu_type={gpu_type}"
        file.write(f"{resource_specification}\n")
        file.write(f"#PBS -lwalltime={walltime}\n")
        # output/error file paths
        file.write(f"#PBS -e {error_path}\n")
        file.write(f"#PBS -o {output_path}\n")
        # initialise conda env
        file.write("module load anaconda3/personal\n")
        file.write(f"source activate {conda_env_name}\n")
        # change to dir where job was submitted from
        file.write("cd $PBS_O_WORKDIR\n")
        # job script
        file.write(f"{run_command}\n")


if __name__ == "__main__":
    args = arg_parser.parse_args()

    create_job_script(
        run_command=args.run_command,
        save_path=args.save_path,
        num_cpus=args.num_cpus,
        conda_env_name=args.conda_env_name,
        memory=args.memory,
        num_gpus=args.num_gpus,
        gpu_type=args.gpu_type,
        error_path=args.error_path,
        output_path=args.output_path,
    )
