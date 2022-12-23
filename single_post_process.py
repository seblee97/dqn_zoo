import argparse
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--results_folder", type=str, help="path to folder containing results."
)
argparser.add_argument("--smoothing", type=int, default=30)


def smooth_data(data: List[float], window_width: int) -> List[float]:
    """Calculates moving average of list of values

    Args:
        data: raw, un-smoothed data.
        window_width: width over which to take moving averags

    Returns:
        smoothed_values: averaged data
    """

    def _smooth(single_dataset):
        cumulative_sum = np.cumsum(single_dataset, dtype=np.float32)
        cumulative_sum[window_width:] = (
            cumulative_sum[window_width:] - cumulative_sum[:-window_width]
        )
        # explicitly forcing data type to 32 bits avoids floating point errors
        # with constant dat.
        smoothed_values = np.array(
            cumulative_sum[window_width - 1 :] / window_width, dtype=np.float32
        )
        return smoothed_values

    if all(isinstance(d, list) for d in data):
        smoothed_data = []
        for dataset in data:
            smoothed_data.append(_smooth(dataset))
    elif all(
        (isinstance(d, float) or isinstance(d, int) or isinstance(d, np.int64))
        for d in data
    ):
        smoothed_data = _smooth(data)

    return smoothed_data


def plot(
    dfs: Dict[str, pd.DataFrame],
    attribute: str,
    algorithm: str,
    environment: str,
    smoothing: int,
    save_path: str,
):
    # organise data to average over seeds
    seeded_data = {}

    for df_name, df in dfs.items():
        config = df_name.strip(".csv").split(f"{algorithm}_{environment}_")[1]
        config_split = config.split("_")
        seed = config_split[-1]
        penalty_strength = config_split[-2]
        penalty_type = "_".join(config_split[:-2])

        if not f"{penalty_type}_{penalty_strength}" in seeded_data:
            seeded_data[f"{penalty_type}_{penalty_strength}"] = []
        seeded_data[f"{penalty_type}_{penalty_strength}"].append(df)

    fig = plt.figure()

    for exp_config, all_seed_dfs in seeded_data.items():
        attribute_data = [df[attribute].dropna() for df in all_seed_dfs]

        if smoothing is not None:
            attribute_data = [
                smooth_data(np.array(data), smoothing) for data in attribute_data
            ]
            min_max = min([len(data) for data in attribute_data])
            attribute_data = [data[:min_max] for data in attribute_data]

        mean_attribute_data = np.mean(attribute_data, axis=0)
        std_attribute_data = np.std(attribute_data, axis=0)

        mean_attribute_data = mean_attribute_data[~np.isnan(mean_attribute_data)]
        std_attribute_data = std_attribute_data[~np.isnan(std_attribute_data)]

        plt.plot(range(len(mean_attribute_data)), mean_attribute_data, label=exp_config)
        plt.fill_between(
            range(len(mean_attribute_data)),
            mean_attribute_data - std_attribute_data,
            mean_attribute_data + std_attribute_data,
            alpha=0.4,
        )

    plt.legend()

    fig.savefig(save_path, dpi=100)


if __name__ == "__main__":

    args = argparser.parse_args()

    attributes = [
        "train_episode_return",
        "eval_episode_return",
        "train_loss",
    ]

    os.makedirs(os.path.join(args.results_folder, "plots"), exist_ok=True)
    df = pd.read_csv(os.path.join(args.results_folder, "writer.csv"))

    for attribute in attributes:
        fig = plt.figure()
        df[attribute].plot()
        plt.xlabel("steps")
        plt.ylabel(attribute)
        fig.savefig(os.path.join(args.results_folder, "plots", f"{attribute}_plot.pdf"))

    nrows = 3
    ncols = 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

    a_index = 0
    for row in range(nrows):
        for col in range(ncols):
            if a_index < len(attributes):
                df[attributes[a_index]].plot(ax=ax[row, col])
                ax[row, col].set_xlabel("step")
                ax[row, col].set_ylabel(attributes[a_index])
                a_index += 1

    fig.savefig(os.path.join(args.results_folder, "plots", f"all_plot.pdf"))
