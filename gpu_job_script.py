import argparse
import datetime
import os
import time

parser = argparse.ArgumentParser()

parser.add_argument("--num_nodes", default=1)
parser.add_argument("--num_gpus_per_node", default=1)
parser.add_argument("--mem", default=32)
parser.add_argument("--timeout", default="0-24:00")
parser.add_argument("--dqn_algorithm", default="bootstrapped_dqn")


def _generate_script(num_nodes: int, num_gpus: int, mem: int, timeout: str, algo: str):

    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    exp_path = os.path.join("results", exp_timestamp)
    os.makedirs(exp_path, exist_ok=True)

    output_path = os.path.join(exp_path, "output.txt")
    error_path = os.path.join(exp_path, "error.txt")

    with open("script", "+w") as script_file:
        script_file.write("#!/bin/bash\n")
        script_file.write("#SBATCH -p gpu\n")
        script_file.write(f"#SBATCH -N {num_nodes}\n")
        script_file.write(f"#SBATCH --mem {mem}G\n")
        script_file.write(f"#SBATCH --gres=gpu:{num_gpus}\n")
        script_file.write(f"#SBATCH --gpus-per-node={num_gpus}\n")
        script_file.write(f"#SBATCH --time={timeout}\n")
        script_file.write(f"#SBATCH --output={output_path}\n")
        script_file.write(f"#SBATCH --error={error_path}\n")
        script_file.write(
            f"python -m dqn_zoo.{algo}.run_key_door --results_path={exp_path}\n"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    _generate_script(
        num_nodes=args.num_nodes,
        num_gpus=args.num_gpus_per_node,
        mem=args.mem,
        timeout=args.timeout,
        algo=args.dqn_algorithm,
    )
