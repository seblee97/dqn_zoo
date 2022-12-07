import itertools
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from dqn_zoo.key_door import constants


def parse_map_outline(map_file_path: str, mapping: Dict[str, int]) -> np.ndarray:
    """Method to parse ascii map and map settings from yaml file.

    Args:
        map_file_path: path to file containing map schematic.
        map_yaml_path: path to yaml file containing map config.

    Returns:
        multi_room_grid: numpy array of map state.
    """
    map_rows = []

    with open(map_file_path) as f:
        map_lines = f.read().splitlines()

        # flip indices for x, y referencing
        for i, line in enumerate(map_lines[::-1]):
            map_row = [mapping[char] for char in line]
            map_rows.append(map_row)

    assert all(
        len(i) == len(map_rows[0]) for i in map_rows
    ), "ASCII map must specify rectangular grid."

    multi_room_grid = np.array(map_rows, dtype=float)

    return multi_room_grid


def parse_x_positions(map_yaml_path: str, data_key: str):
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    positions = [tuple(p) for p in map_data[data_key]]

    return positions


def parse_map_positions(map_yaml_path: str) -> Tuple[List, List, List, List]:
    """Method to parse map settings from yaml file.

    Args:
        map_yaml_path: path to yaml file containing map config.

    Returns:
        initial_start_position: x,y coordinates for
            agent at start of each episode.
        key_positions: list of x, y coordinates of keys.
        door_positions: list of x, y coordinates of doors.
        reward_positions: list of x, y coordinates of rewards.
    """
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    start_positions = [tuple(map_data[constants.START_POSITION])]

    reward_positions = parse_x_positions(
        map_yaml_path=map_yaml_path, data_key=constants.REWARD_POSITIONS
    )
    key_positions = parse_x_positions(
        map_yaml_path=map_yaml_path, data_key=constants.KEY_POSITIONS
    )
    door_positions = parse_x_positions(
        map_yaml_path=map_yaml_path, data_key=constants.DOOR_POSITIONS
    )

    reward_statistics = map_data[constants.REWARD_STATISTICS]

    assert (
        len(start_positions) == 1
    ), "maximally one start position 'S' should be specified in ASCII map."

    assert len(door_positions) == len(
        key_positions
    ), "number of key positions must equal number of door positions."

    return (
        start_positions[0],
        key_positions,
        door_positions,
        reward_positions,
        reward_statistics,
    )


def random_correct_keys(self, num_keys):
    """Returns list of length total_rewards (equivalently number of each key)
    such that each index is 0 or 1 with equal probability."""
    return [0 if s < 0.5 else 1 for s in np.random.random(size=num_keys)]


def parse_posner_map_positions(map_yaml_path: str) -> Tuple[List, List, List, List]:
    """Method to parse map settings from yaml file.

    Args:
        map_yaml_path: path to yaml file containing map config.

    Returns:
        initial_start_position: x,y coordinates for
            agent at start of each episode.
        key_positions: list of x, y coordinates of keys.
        door_positions: list of x, y coordinates of doors.
        reward_positions: list of x, y coordinates of rewards.
    """
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    start_positions = [tuple(map_data[constants.START_POSITION])]

    reward_by = map_data[constants.REWARD_BY]
    keys_change_color = map_data[constants.KEYS_CHANGE_COLOR]

    if map_data.get(f"{constants.SILVER}_{constants.KEY_POSITIONS}"):
        silver_key_positions = {
            i: pos
            for i, pos in enumerate(
                parse_x_positions(
                    map_yaml_path=map_yaml_path,
                    data_key=f"{constants.SILVER}_{constants.KEY_POSITIONS}",
                )
            )
        }
    else:
        silver_key_positions = None

    if map_data.get(f"{constants.GOLD}_{constants.KEY_POSITIONS}"):
        gold_key_positions = {
            i: pos
            for i, pos in enumerate(
                parse_x_positions(
                    map_yaml_path=map_yaml_path,
                    data_key=f"{constants.GOLD}_{constants.KEY_POSITIONS}",
                )
            )
        }
    else:
        gold_key_positions = None

    key_1_positions = {
        i: pos
        for i, pos in enumerate(
            parse_x_positions(
                map_yaml_path=map_yaml_path,
                data_key=f"{constants.KEY}_1_{constants.POSITIONS}",
            )
        )
    }

    key_2_positions = {
        i: pos
        for i, pos in enumerate(
            parse_x_positions(
                map_yaml_path=map_yaml_path,
                data_key=f"{constants.KEY}_2_{constants.POSITIONS}",
            )
        )
    }

    correct_keys = map_data.get(constants.CORRECT_KEYS)

    correct_keys_change = map_data[constants.CORRECT_KEYS_CHANGE]

    door_positions = parse_x_positions(
        map_yaml_path=map_yaml_path,
        data_key=constants.DOOR_POSITIONS,
    )

    reward_positions = parse_x_positions(
        map_yaml_path=map_yaml_path,
        data_key=constants.REWARD_POSITIONS,
    )

    reward_statistics = map_data[constants.REWARD_STATISTICS]
    cue_specification = {
        k: map_data.get(k)
        for k in [
            constants.CUE_FORMAT,
            constants.CUE_VALIDITY,
            constants.CUE_SIZE,
            constants.NUM_CUES,
            constants.CUE_LINE_DEPTH,
        ]
    }

    assert (
        len(start_positions) == 1
    ), "maximally one start position 'S' should be specified in ASCII map."

    assert len(door_positions) == len(
        key_1_positions
    ), "number of key positions must equal number of door positions."

    assert len(door_positions) == len(
        key_2_positions
    ), "number of key positions must equal number of door positions."

    return (
        start_positions[0],
        reward_by,
        silver_key_positions,
        gold_key_positions,
        key_1_positions,
        key_2_positions,
        correct_keys,
        keys_change_color,
        correct_keys_change,
        door_positions,
        reward_positions,
        reward_statistics,
        cue_specification,
    )


def parse_spatial_keys_maze_map_positions(
    map_yaml_path: str,
) -> Tuple[List, List, List, List]:
    """Method to parse map settings from yaml file.

    Args:
        map_yaml_path: path to yaml file containing map config.

    Returns:
        initial_start_position: x,y coordinates for
            agent at start of each episode.
        key_positions: list of x, y coordinates of keys.
        door_positions: list of x, y coordinates of doors.
        reward_positions: list of x, y coordinates of rewards.
    """
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    start_positions = [tuple(map_data[constants.START_POSITION])]

    key_1_positions = parse_x_positions(
        map_yaml_path=map_yaml_path,
        data_key=f"{constants.KEY}_1_{constants.POSITIONS}",
    )

    key_2_positions = parse_x_positions(
        map_yaml_path=map_yaml_path,
        data_key=f"{constants.KEY}_2_{constants.POSITIONS}",
    )

    correct_keys_change = map_data[constants.CORRECT_KEYS_CHANGE]
    color_change = map_data[constants.COLOR_CHANGE]

    door_positions = parse_x_positions(
        map_yaml_path=map_yaml_path,
        data_key=constants.DOOR_POSITIONS,
    )

    reward_positions = parse_x_positions(
        map_yaml_path=map_yaml_path,
        data_key=constants.REWARD_POSITIONS,
    )

    reward_statistics = map_data[constants.REWARD_STATISTICS]

    assert (
        len(start_positions) == 1
    ), "maximally one start position 'S' should be specified in ASCII map."

    assert len(door_positions) == len(
        key_1_positions
    ), "number of key positions must equal number of door positions."

    assert len(door_positions) == len(
        key_2_positions
    ), "number of key positions must equal number of door positions."

    return (
        start_positions[0],
        key_1_positions,
        key_2_positions,
        correct_keys_change,
        color_change,
        door_positions,
        reward_positions,
        reward_statistics,
    )


def parse_weinan_map_positions(map_yaml_path: str) -> Tuple[List, List, List, List]:
    """Method to parse map settings from yaml file.

    Args:
        map_yaml_path: path to yaml file containing map config.

    Returns:
        initial_start_position: x,y coordinates for
            agent at start of each episode.
        key_positions: list of x, y coordinates of keys.
        door_positions: list of x, y coordinates of doors.
        reward_positions: list of x, y coordinates of rewards.
    """
    with open(map_yaml_path) as yaml_file:
        map_data = yaml.load(yaml_file, yaml.SafeLoader)

    import pdb

    pdb.set_trace()

    start_positions = [tuple(map_data[constants.START_POSITION])]

    cue_positions = [tuple(map_data[constants.CUE_POSITION])]

    reward_positions = parse_x_positions(
        map_yaml_path=map_yaml_path,
        data_key=constants.REWARD_POSITIONS,
    )

    reward_statistics = map_data[constants.REWARD_STATISTICS]

    assert (
        len(start_positions) == 1
    ), "maximally one start position 'S' should be specified in ASCII map."

    return (start_positions[0], cue_positions[0], reward_positions, reward_statistics)


def setup_reward_statistics(
    reward_positions, reward_specifications
) -> Dict[Tuple, Callable]:

    per_reward_specification = len(reward_positions) == len(reward_specifications)
    single_reward_specification = len(reward_specifications) == 1

    assert (
        per_reward_specification or single_reward_specification
    ), "number of rewards statistics must either be 1 or match number of reward positions."

    def _get_reward_function(reward_type: str, reward_parameters: Dict) -> Callable:

        if reward_type == constants.GAUSSIAN:

            def _sample_gaussian():
                return np.random.normal(
                    loc=reward_parameters[constants.MEAN],
                    scale=reward_parameters[constants.VARIANCE],
                )

            return _sample_gaussian

    reward_types = list(reward_specifications.keys())
    reward_parameters = list(reward_specifications.values())

    if single_reward_specification:
        reward_function = _get_reward_function(reward_types[0], reward_parameters[0])
        rewards = {
            reward_position: reward_function for reward_position in reward_positions
        }
    else:
        rewards = {
            reward_position: _get_reward_function(reward_type, reward_parameter)
            for reward_position, reward_type, reward_parameter in zip(
                reward_positions, reward_types, reward_parameters
            )
        }

    return rewards


def configure_state_space(map_outline, key_positions, reward_positions):
    """Get state space for the environment from the parsed map.
    Further split state space into walls, valid positions, key possessions etc.
    """
    state_indices = np.where(map_outline == 0)
    wall_indices = np.where(map_outline == 1)

    positional_state_space = list(zip(state_indices[1], state_indices[0]))
    key_possession_state_space = list(
        itertools.product([0, 1], repeat=len(key_positions))
    )
    rewards_received_state_space = list(
        itertools.product([0, 1], repeat=len(reward_positions))
    )
    state_space = [
        i[0] + i[1] + i[2]
        for i in itertools.product(
            positional_state_space,
            key_possession_state_space,
            rewards_received_state_space,
        )
    ]

    wall_state_space = list(zip(wall_indices[1], wall_indices[0]))

    return (
        positional_state_space,
        key_possession_state_space,
        rewards_received_state_space,
        state_space,
        wall_state_space,
    )


def configure_posner_state_space(
    map_outline,
    key_1_positions,
    key_2_positions,
    reward_positions,
):
    """Get state space for the environment from the parsed map.
    Further split state space into walls, valid positions, key possessions etc.
    """
    state_indices = np.where(map_outline == 0)
    wall_indices = np.where(map_outline == 1)

    positional_state_space = list(zip(state_indices[1], state_indices[0]))
    key_1_possession_state_space = list(
        itertools.product([0, 1], repeat=len(key_1_positions))
    )
    key_2_possession_state_space = list(
        itertools.product([0, 1], repeat=len(key_2_positions))
    )
    rewards_received_state_space = list(
        itertools.product([0, 1], repeat=len(reward_positions))
    )
    state_space = [
        i[0] + i[1] + i[2] + i[3]
        for i in itertools.product(
            positional_state_space,
            key_1_possession_state_space,
            key_2_possession_state_space,
            rewards_received_state_space,
        )
    ]

    wall_state_space = list(zip(wall_indices[1], wall_indices[0]))

    return (
        positional_state_space,
        key_1_possession_state_space,
        key_2_possession_state_space,
        rewards_received_state_space,
        state_space,
        wall_state_space,
    )


def configure_weinan_state_space(
    map_outline,
    reward_positions,
):
    """Get state space for the environment from the parsed map.
    Further split state space into walls, valid positions, key possessions etc.
    """
    state_indices = np.where(map_outline == 0)
    wall_indices = np.where(map_outline == 1)

    positional_state_space = list(zip(state_indices[1], state_indices[0]))
    rewards_received_state_space = list(
        itertools.product([0, 1], repeat=len(reward_positions))
    )
    state_space = [
        i[0] + i[1]
        for i in itertools.product(
            positional_state_space,
            rewards_received_state_space,
        )
    ]

    wall_state_space = list(zip(wall_indices[1], wall_indices[0]))

    return (
        positional_state_space,
        rewards_received_state_space,
        state_space,
        wall_state_space,
    )


def rgb_to_grayscale(rgb: np.ndarray) -> np.ndarray:
    # rgb channel last
    grayscale = np.dot(rgb[..., :3], [[0.299], [0.587], [0.114]])
    return grayscale
