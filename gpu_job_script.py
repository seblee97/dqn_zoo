import argparse
import datetime
import os
import shutil
import subprocess
import time

parser = argparse.ArgumentParser()

parser.add_argument("--num_nodes", default=1)
parser.add_argument("--num_gpus_per_node", default=1)
parser.add_argument("--mem", default=32)
parser.add_argument("--timeout", default="0-120:00")
parser.add_argument("--game", default="atari")
parser.add_argument("--dqn_algorithm", default="bootstrapped_dqn")


def _generate_script(
    num_nodes: int, num_gpus: int, mem: int, timeout: str, algo: str, game: str
):

    raw_datetime = datetime.datetime.fromtimestamp(time.time())
    exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    exp_path = os.path.join("results", exp_timestamp)
    os.makedirs(exp_path, exist_ok=True)

    output_path = os.path.join(exp_path, "output.txt")
    error_path = os.path.join(exp_path, "error.txt")

    run_command = f"python -m dqn_zoo.{algo}.run_{game} --results_path={exp_path} "

    if game == "key_door":

        key_door_map_path = os.path.join(exp_path, "key_door_maps")
        map_ascii_path = os.path.join(key_door_map_path, "multi_room_bandit.txt")
        map_yaml_path = os.path.join(key_door_map_path, "multi_room_bandit.yaml")
        map_yaml_paths = ",".join(
            [
                os.path.join(key_door_map_path, yml)
                for yml in [
                    "multi_room_bandit.yaml",
                    "multi_room_bandit_1.yaml",
                    "multi_room_bandit_2.yaml",
                ]
            ]
        )
        shutil.copytree("dqn_zoo/key_door_maps", key_door_map_path)

        run_command += (
            f"--map_ascii_path={map_ascii_path} --map_yaml_path={map_yaml_path} "
            f"--map_yaml_paths={map_yaml_paths}"
        )

    script_path = os.path.join(exp_path, "script")

    with open(script_path, "+w") as script_file:
        script_file.write("#!/bin/bash\n")
        script_file.write("#SBATCH -p gpu\n")
        script_file.write(f"#SBATCH -N {num_nodes}\n")
        script_file.write(f"#SBATCH --mem {mem}G\n")
        script_file.write(f"#SBATCH --gres=gpu:{num_gpus}\n")
        script_file.write(f"#SBATCH --gpus-per-node={num_gpus}\n")
        script_file.write(f"#SBATCH --time={timeout}\n")
        script_file.write(f"#SBATCH --output={output_path}\n")
        script_file.write(f"#SBATCH --error={error_path}\n")
        script_file.write(f"{run_command}\n")

    return script_path


if __name__ == "__main__":
    args = parser.parse_args()
    script_path = _generate_script(
        num_nodes=args.num_nodes,
        num_gpus=args.num_gpus_per_node,
        mem=args.mem,
        timeout=args.timeout,
        algo=args.dqn_algorithm,
        game=args.game,
    )
    subprocess.call(f"sbatch {script_path}", shell=True)
