import numpy as np


def hours_to_minutes(time):
    hours = float(time.split(":")[0])
    minutes = float(time.split(":")[1])
    seconds = float(time.split(":")[2])
    return hours * 60 + minutes + seconds / 60

def hours_to_seconds(time):
    hours = float(time.split(":")[0])
    minutes = float(time.split(":")[1])
    seconds = float(time.split(":")[2])
    return hours * 3600 + minutes * 60 + seconds


if __name__ == "__main__":
    file_path = "times_log2"
    list_of_models = ["CPU ENSEMBLE QR-DQN",
                      "CPU QR-DQN",
                      "CPU DQN",
                      "GPU ENSEMBLE QR-DQN",
                      "GPU QR-DQN",
                      "GPU DQN"]
    times_per_model = {}
    for model in list_of_models:
        times_per_model[model] = []

    with open(file_path, "r") as file:
        lines = file.readlines()
        current_model = ""
        for line in lines:
            for model in list_of_models:
                if model in line:
                    current_model = model
                    break
            if "I0810" in line and "Training iteration" in line:
                time = line.split(" ")[1]
                times_per_model[current_model].append(hours_to_seconds(time))

    # Convert everything to numpy array
    for model in list_of_models:
        times_per_model[model] = np.array(times_per_model[model])

    # take the last 20 iterations and compute the mean and std of the derivative per model
    for model in list_of_models:
        times_per_model[model] = times_per_model[model][-21:]
        print(f"Model: {model}")
        print(f"Mean: {np.mean(np.diff(times_per_model[model]))}")
        print(f"Std: {np.std(np.diff(times_per_model[model]))}")
        print(f"Max: {np.max(np.diff(times_per_model[model]))}")
        print(f"Min: {np.min(np.diff(times_per_model[model]))}")
        print("\n")