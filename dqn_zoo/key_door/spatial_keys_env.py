import copy
import itertools
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from dqn_zoo.key_door import base_environment, constants, utils


class SpatialKeysEnv(base_environment.BaseEnvironment):
    """Grid world environment with multiple rooms.
    Between each room is a door, that requires a key to unlock.

    There are two keys of different colours in each room. The colour
    of the key is irrelevant, only the position is important.
    Each key will open the door, but if the incorrect key is collected
    first, the reward in the following room disappears.
    """

    def __init__(
        self,
        map_ascii_path: str,
        map_yaml_path: str,
        representation: str,
        episode_timeout: Optional[Union[int, None]] = None,
        frame_stack: Optional[Union[int, None]] = None,
        scaling: Optional[int] = 1,
        field_x: Optional[int] = 1,
        field_y: Optional[int] = 1,
        grayscale: bool = True,
        batch_dimension: bool = True,
        torch_axes: bool = True,
    ) -> None:
        """Class constructor.

        Args:
            map_ascii_path: path to txt or other ascii file with map specifications.
            map_yaml_path: path to yaml file with map settings (reward locations etc.)
            representation: agent_position (for tabular) or pixel
                (for function approximation).
            episode_timeout: number of steps before episode automatically terminates.
            scaling: optional integer (for use with pixel representations)
                specifying how much to expand state by.
            field_x: integer (required for use with partial observability
                in pixel representations) specifying how many pixels in each x
                direction the agent can see.
            field_y: integer (required for use with partial observability
                in pixel representations) specifying how many pixels in each y
                direction the agent can see.
            grayscale: whether to grayscale representation.
            batch_dimension: whether to add dummy batch dimension to representation.
            torch_axes: whether to use torch (else tf) axis ordering.
        """

        self._key_ids = [constants.GOLD, constants.SILVER]

        super().__init__(
            representation=representation,
            episode_timeout=episode_timeout,
            frame_stack=frame_stack,
            scaling=scaling,
            field_x=field_x,
            field_y=field_y,
            grayscale=grayscale,
            batch_dimension=batch_dimension,
            torch_axes=torch_axes,
        )

        self._setup_environment(
            map_ascii_path=map_ascii_path, map_yaml_path=map_yaml_path
        )

        # states are zero, -1 removes walls from counts.
        self._visitation_counts = -1 * copy.deepcopy(self._map)

    def _setup_environment(
        self, map_yaml_path: str, map_ascii_path: Optional[str] = None
    ):
        """Setup environment according to geometry of ascii file
        and settings of yaml file.

        Args:
            map_ascii_path: path to txt or other ascii file with map specifications.
            map_yaml_path: path to yaml file with map settings (reward locations etc.)
        """

        if map_ascii_path is not None:
            self._map = utils.parse_map_outline(
                map_file_path=map_ascii_path, mapping=self.MAPPING
            )

        (
            self._starting_xy,
            self._key_1_positions,
            self._key_2_positions,
            self._correct_keys_change,
            self._color_change,
            self._door_positions,
            reward_positions,
            reward_statistics,
        ) = utils.parse_spatial_keys_maze_map_positions(map_yaml_path)

        self._rewards = utils.setup_reward_statistics(
            reward_positions, reward_statistics
        )

        self._total_rewards = len(self._rewards)
        self._accessible_rewards = len(self._rewards)

        gold_indices = np.random.binomial(1, 0.5, len(self._key_1_positions))
        self._gold_key_positions = [
            self._key_1_positions[i] if gold_indices[i] else self._key_2_positions[i]
            for i in range(len(gold_indices))
        ]
        self._silver_key_positions = [
            self._key_2_positions[i] if gold_indices[i] else self._key_1_positions[i]
            for i in range(len(gold_indices))
        ]

        if self._correct_keys_change:
            pass
        else:
            correct_indices = np.random.binomial(1, 0.5, len(self._key_1_positions))
            self._correct_keys = [
                self._key_1_positions[i]
                if correct_indices[i]
                else self._key_2_positions[i]
                for i in range(len(correct_indices))
            ]

        (
            self._positional_state_space,
            self._silver_key_possession_state_space,
            self._gold_key_possession_state_space,
            self._rewards_received_state_space,
            self._state_space,
            self._wall_state_space,
        ) = utils.configure_posner_state_space(
            map_outline=self._map,
            silver_key_positions=self._silver_key_positions,
            gold_key_positions=self._gold_key_positions,
            reward_positions=reward_positions,
        )

    def average_values_over_positional_states(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], float]:
        """For certain analyses (e.g. plotting value functions) we want to
        average the values for each position over all non-positional state information--
        in this case the key posessions.
        Args:
            values: full state-action value information
        Returns:
            averaged_values: (positional-state)-action values.
        """
        averaged_values = {}
        for state in self._positional_state_space:
            non_positional_set = [
                values[state + i[0] + i[1] + i[2]]
                for i in itertools.product(
                    self._silver_key_possession_state_space,
                    self._gold_key_possession_state_space,
                    self._rewards_received_state_space,
                )
            ]
            non_positional_mean = np.mean(non_positional_set, axis=0)
            averaged_values[state] = non_positional_mean
        return averaged_values

    def get_value_combinations(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], Dict[Tuple[int], float]]:
        """Get each possible combination of positional state-values
        over non-positional states.
        Args:
            values: values over overall state-space
        Returns:
            value_combinations: values over positional state space
                for each combination of non-positional state.
        """
        value_combinations = {}
        for silver_key_state in self._silver_key_possession_state_space:
            for gold_key_state in self._gold_key_possession_state_space:
                for reward_state in self._rewards_received_state_space:
                    value_combination = {}
                    for state in self._positional_state_space:
                        value_combination[state] = values[
                            state + silver_key_state + gold_key_state + reward_state
                        ]
                    value_combinations[
                        silver_key_state + gold_key_state + reward_state
                    ] = value_combination

        return value_combinations

    def _env_skeleton(
        self,
        rewards: Union[None, str, Tuple[int]] = "state",
        keys: Dict[str, Union[None, str, Tuple[int]]] = {
            "gold": "state",
            "silver": "state",
        },
        doors: Union[None, str] = "state",
        agent: Union[None, str, np.ndarray] = "state",
        cue: Union[None, str, np.ndarray] = "state",
    ) -> np.ndarray:
        """Get a 'skeleton' of map e.g. for visualisation purposes.

        Args:
            rewards: # TODO whether or not to mark out rewards (ignores magnitudes).
            show_doors: whether or not to mark out doors.
            show_keys: whether or not to mark out keys.
            show_agent: whether or not to mark out agent position

        Returns:
            skeleton: np array of map.
        """
        # flip size so indexing is consistent with axis dimensions
        skeleton = np.ones(self._map.shape + (3,))

        # make walls black
        skeleton[self._map == 1] = np.zeros(3)

        if rewards is not None:
            if isinstance(rewards, str):
                if rewards == constants.STATIONARY:
                    reward_iterate = list(self._rewards.keys())
                elif rewards == constants.STATE:
                    reward_positions = list(self._rewards.keys())

                    wrong_keys_collected = []
                    for i, k in enumerate(self._correct_keys):
                        if k in self._gold_key_positions and self._silver_keys_state[i]:
                            wrong_keys_collected.append(True)
                        elif (
                            k in self._silver_key_positions and self._gold_keys_state[i]
                        ):
                            wrong_keys_collected.append(True)
                        else:
                            wrong_keys_collected.append(False)

                    self._accessible_rewards = len(self._rewards) - sum(
                        wrong_keys_collected
                    )

                    reward_iterate = [
                        reward_positions[i]
                        for i, r in enumerate(self._rewards_state)
                        if not r and not wrong_keys_collected[i]
                    ]
                else:
                    raise ValueError(f"Rewards keyword {rewards} not identified.")
            elif isinstance(rewards, tuple):
                reward_positions = list(self._rewards.keys())
                reward_iterate = [
                    reward_positions[i] for i, r in enumerate(rewards) if not r
                ]
            else:
                raise ValueError(
                    "Rewards must be string with relevant keyword or keystate list"
                )
            # show reward in green
            for reward in reward_iterate:
                skeleton[reward[::-1]] = [0.0, 0.5, 0.0]

        silver_keys = keys[constants.SILVER]
        if silver_keys is not None:
            if isinstance(silver_keys, str):
                if silver_keys == constants.STATIONARY:
                    silver_keys_iterate = self._silver_key_positions
                elif silver_keys == constants.STATE:
                    silver_keys_iterate = [
                        self._silver_key_positions[i]
                        for i, k in enumerate(self._silver_keys_state)
                        if (not self._gold_keys_state[i] and not k)
                    ]
                else:
                    raise ValueError(f"Keys keyword {silver_keys} not identified.")
            elif isinstance(silver_keys, tuple):
                silver_keys_iterate = [
                    self._silver_key_positions[i]
                    for i, k in enumerate(silver_keys)
                    if not k
                ]
            else:
                raise ValueError(
                    "Keys must be string with relevant keyword or keystate list"
                )

            # show silver key in silver
            for key_position in silver_keys_iterate:
                skeleton[tuple(key_position[::-1])] = [0.753, 0.753, 0.753]

        gold_keys = keys[constants.GOLD]
        if gold_keys is not None:
            if isinstance(gold_keys, str):
                if gold_keys == constants.STATIONARY:
                    gold_keys_iterate = self._gold_key_positions
                elif gold_keys == constants.STATE:
                    gold_keys_iterate = [
                        self._gold_key_positions[i]
                        for i, k in enumerate(self._gold_keys_state)
                        if (not self._silver_keys_state[i] and not k)
                    ]
                else:
                    raise ValueError(f"Keys keyword {gold_keys} not identified.")
            elif isinstance(gold_keys, tuple):
                gold_keys_iterate = [
                    self._gold_key_positions[i]
                    for i, k in enumerate(gold_keys)
                    if not k
                ]
            else:
                raise ValueError(
                    "Keys must be string with relevant keyword or keystate list"
                )

            # show gold key in gold
            for key_position in gold_keys_iterate:
                skeleton[tuple(key_position[::-1])] = [1.0, 0.843, 0.0]

        if doors is not None:
            if isinstance(doors, str):
                if doors == constants.STATE:
                    doors_iterate = self._door_positions
                elif doors == constants.STATIONARY:
                    doors_iterate = self._door_positions
                else:
                    raise ValueError(f"Doors keyword {doors} not identified.")
            # show door in red
            for door in doors_iterate:
                skeleton[tuple(door[::-1])] = [1.0, 0.0, 0.0]

        if agent is not None:
            if isinstance(agent, str):
                if agent == constants.STATE:
                    agent_position = self._agent_position
                elif agent == constants.STATIONARY:
                    agent_position = self._starting_xy
            else:
                agent_position = agent
            # show agent in blue
            skeleton[tuple(agent_position[::-1])] = [0.0, 0.0, 1.0]

        if cue is not None:
            if isinstance(cue, str):
                if cue == constants.STATE:
                    cue_line = np.tile(
                        constants.BLACK_PIXEL,
                        [1, skeleton.shape[1], 1],
                    )

                    skeleton = np.vstack((cue_line, skeleton))

        return skeleton

    def _partial_observation(self, state, agent_position):

        height = state.shape[0]
        width = state.shape[1]

        # out of bounds needs to be different from wall pixels
        OUT_OF_BOUNDS_PIXEL = 0.2 * np.ones(3)

        # nominal bounds on field of view (pre-edge cases)
        x_min = agent_position[1] - self._field_x
        x_max = agent_position[1] + self._field_x
        y_min = agent_position[0] - self._field_y
        y_max = agent_position[0] + self._field_y

        state = state[
            max(0, x_min) : min(x_max, width) + 1,
            max(0, y_min) : min(y_max, height) + 1,
            :,
        ]

        # edge case contingencies
        if 0 > x_min:
            append_left = 0 - x_min

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((append_left, state.shape[1], 1)),
            )
            state = np.concatenate(
                (fill, state),
                axis=0,
            )
        if x_max >= width:
            append_right = x_max + 1 - width

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((append_right, state.shape[1], 1)),
            )
            state = np.concatenate(
                (state, fill),
                axis=0,
            )
        if 0 > y_min:
            append_below = 0 - y_min

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((state.shape[0], append_below, 1)),
            )
            state = np.concatenate(
                (fill, state),
                axis=1,
            )
        if y_max >= height:
            append_above = y_max + 1 - height

            fill = np.kron(
                OUT_OF_BOUNDS_PIXEL,
                np.ones((state.shape[0], append_above, 1)),
            )
            state = np.concatenate(
                (state, fill),
                axis=1,
            )

        return state

    def get_state_representation(
        self,
        tuple_state: Optional[Tuple] = None,
    ) -> Union[tuple, np.ndarray]:
        """From current state, produce a representation of it.
        This can either be a tuple of the agent and key positions,
        or a top-down pixel view of the environment (for DL)."""
        if self._representation == constants.AGENT_POSITION:
            return (
                tuple(self._agent_position)
                + tuple(self._silver_keys_state)
                + tuple(self._gold_keys_state)
                + tuple(self._rewards_state)
            )
        elif self._representation in [constants.PIXEL, constants.PO_PIXEL]:
            if tuple_state is None:
                agent_position = self._agent_position
                state = self._env_skeleton()  # H x W x C
            else:
                agent_position = tuple_state[:2]
                silver_keys = tuple_state[2 : 2 + len(self._silver_key_positions)]
                gold_keys = tuple_state[2 : 2 + len(self._gold_key_positions)]
                rewards = tuple_state[
                    2
                    + len(self._silver_key_positions)
                    + len(self._gold_key_positions) :
                ]
                state = self._env_skeleton(
                    rewards=rewards,
                    keys={constants.SILVER: silver_keys, constants.GOLD: gold_keys},
                    agent=agent_position,
                )  # H x W x C

            if self._representation == constants.PO_PIXEL:
                state = self._partial_observation(
                    state=state, agent_position=agent_position
                )

            if self._grayscale:
                state = utils.rgb_to_grayscale(state)

            if self._torch_axes:
                state = np.transpose(state, axes=(2, 0, 1))  # C x H x W
                state = np.kron(state, np.ones((1, self._scaling, self._scaling)))
            else:
                state = np.kron(state, np.ones((self._scaling, self._scaling, 1)))

            if self._batch_dimension:
                # add batch dimension
                state = np.expand_dims(state, 0)

            return state

    def _move_agent(self, delta: np.ndarray) -> float:
        """Move agent. If provisional new position is a wall, no-op."""
        provisional_new_position = self._agent_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._wall_state_space
        locked_door = tuple(provisional_new_position) in self._door_positions

        if locked_door:
            door_index = self._door_positions.index(tuple(provisional_new_position))
            if self._gold_keys_state[door_index] or self._silver_keys_state[door_index]:
                locked_door = False

        if not moving_into_wall and not locked_door:
            self._agent_position = provisional_new_position

        if tuple(self._agent_position) in self._silver_key_positions:
            key_index = self._silver_key_positions.index(tuple(self._agent_position))
            if (
                not self._silver_keys_state[key_index]
                and not self._gold_keys_state[key_index]
            ):
                self._silver_keys_state[key_index] = 1
        if tuple(self._agent_position) in self._gold_key_positions:
            key_index = self._gold_key_positions.index(tuple(self._agent_position))
            if (
                not self._gold_keys_state[key_index]
                and not self._silver_keys_state[key_index]
            ):
                self._gold_keys_state[key_index] = 1

        return self._compute_reward()

    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent.

        Args:
            action: 0: left, 1: up, 2: right, 3: down

        Returns:
            reward: float indicating reward, 1 for target reached, 0 otherwise.
            next_state: new coordinates of agent.
        """
        assert (
            self._active
        ), "Environment not active. call reset_environment() to reset environment and make it active."
        assert (
            action in self.ACTION_SPACE
        ), f"Action given as {action}; must be 0: left, 1: up, 2: right or 3: down."

        reward = self._move_agent(delta=self.DELTAS[action])
        new_state = self.get_state_representation()
        skeleton = self._env_skeleton()

        self._active = self._remain_active(reward=reward)
        self._episode_step_count += 1

        if self._training:
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            self._train_episode_position_history.append(tuple(self._agent_position))
            self._train_episode_history.append(skeleton)
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history.append(
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                )
        else:
            self._test_episode_position_history.append(tuple(self._agent_position))
            self._test_episode_history.append(skeleton)
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history.append(
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                )

        return reward, new_state

    def _compute_reward(self) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log.
        """
        if (
            tuple(self._agent_position) in self._rewards
            and tuple(self._agent_position) not in self._rewards_received
        ):
            reward = self._rewards.get(tuple(self._agent_position))()
            reward_index = list(self._rewards.keys()).index(tuple(self._agent_position))
            self._rewards_state[reward_index] = 1
            self._rewards_received.append(tuple(self._agent_position))
        else:
            reward = 0.0

        return reward

    def _remain_active(self, reward: float) -> bool:
        """Check on reward / timeout conditions whether episode should be terminated.

        Args:
            reward: total reward accumulated so far.

        Returns:
            remain_active: whether to keep episode active.
        """
        conditions = [
            self._episode_step_count == self._episode_timeout,
            len(self._rewards_received) == self._accessible_rewards,
        ]
        return not any(conditions)

    def reset_environment(
        self, train: bool = True, map_yaml_path: Optional[str] = None
    ) -> Tuple[int, int, int]:
        """Reset environment.

        Bring agent back to starting position.

        Args:
            train: whether episode is for train or test (affects logging).
        """
        if map_yaml_path is not None:
            self._setup_environment(map_yaml_path=map_yaml_path)

        self._active = True
        self._episode_step_count = 0
        self._training = train
        self._agent_position = np.array(self._starting_xy)
        self._rewards_received = []
        self._silver_keys_state = np.zeros(len(self._silver_key_positions), dtype=int)
        self._gold_keys_state = np.zeros(len(self._gold_key_positions), dtype=int)
        self._rewards_state = np.zeros(len(self._rewards), dtype=int)

        if self._color_change:
            gold_indices = np.random.binomial(1, 0.5, len(self._key_1_positions))
            self._gold_key_positions = [
                self._key_1_positions[i]
                if gold_indices[i]
                else self._key_2_positions[i]
                for i in range(len(gold_indices))
            ]
            self._silver_key_positions = [
                self._key_2_positions[i]
                if gold_indices[i]
                else self._key_1_positions[i]
                for i in range(len(gold_indices))
            ]

        if self._correct_keys_change:
            correct_indices = np.random.binomial(1, 0.5, len(self._key_1_positions))
            self._correct_keys = [
                self._key_1_positions[i]
                if correct_indices[i]
                else self._key_2_positions[i]
                for i in range(len(correct_indices))
            ]

        initial_state = self.get_state_representation()
        skeleton = self._env_skeleton()

        if train:
            self._train_episode_position_history = [tuple(self._agent_position)]
            self._train_episode_history = [skeleton]
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history = [
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                ]
        else:
            self._test_episode_position_history = [tuple(self._agent_position)]
            self._test_episode_history = [skeleton]
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history = [
                    self._partial_observation(
                        state=skeleton, agent_position=self._agent_position
                    )
                ]

        return initial_state
