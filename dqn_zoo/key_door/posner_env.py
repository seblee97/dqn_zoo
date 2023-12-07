import copy
import itertools
from multiprocessing.sharedctypes import Value
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib.rcsetup import validate_color_or_auto

from dqn_zoo.key_door import base_environment, constants, utils


class PosnerEnv(base_environment.BaseEnvironment):
    """Grid world environment with multiple rooms.
    Between each room is a door, that requires a key to unlock.

    There are two keys of different colours in each room.
    Each key will open the door, but if the incorrect key is collected
    first, the reward in the following room disappears. The correct key
    may be indicated by a visual cue that lines the bottom of the map.
    """

    BLUE_RGB = [0.0, 0.0, 1.0]
    GREEN_RGB = [0.0, 0.5, 0.0]
    RED_RGB = [1.0, 0.0, 0.0]
    GOLD_RGB = [1.0, 0.843, 0.0]
    SILVER_RGB = [0.753, 0.753, 0.753]
    BLACK_RGB = [0.0, 0.0, 0.0]
    WHITE_RGB = [1.0, 1.0, 1.0]

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
            frame_stack: number of frames to stack together in representation.
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
        self._state_visitation_counts = {state: 0 for state in self._state_space}

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
            self._skeleton_shape = self._map.shape + (3,)

        (
            self._starting_xy,
            self._reward_by,
            silver_key_positions,
            gold_key_positions,
            self._key_1_positions,
            self._key_2_positions,
            self._correct_keys,
            self._keys_change_color,
            self._correct_keys_change,
            self._door_positions,
            self._reward_positions,
            reward_statistics,
            self._cue_specification,
        ) = utils.parse_posner_map_positions(map_yaml_path)

        self._key_1_position_index_mapping = {
            pos: i for i, pos in self._key_1_positions.items()
        }
        self._key_2_position_index_mapping = {
            pos: i for i, pos in self._key_2_positions.items()
        }

        self._cue_format = self._cue_specification[constants.CUE_FORMAT]
        if self._cue_format is not None:
            self._cue_validity = self._cue_specification[constants.CUE_VALIDITY]
            self._cue_line_depth = self._cue_specification[constants.CUE_LINE_DEPTH]
            if self._cue_format == constants.POSNER:
                self._cue_size = self._cue_specification[constants.CUE_SIZE]
                self._num_cues = self._cue_specification[constants.NUM_CUES]
                self._cue_index = self._cue_specification.get(
                    constants.CUE_INDEX, np.random.choice(self._num_cues)
                )

                assert (
                    self._cue_size * self._num_cues < self._map.shape[1]
                ), "cue line (size of cue * number of cues) must be less than width of map"
        else:
            self._cue_line_depth = self._cue_specification[constants.CUE_LINE_DEPTH]
        self._default_cue_line = np.tile(
            self.BLACK_RGB,
            [self._cue_line_depth, self._skeleton_shape[1], 1],
        )

        self._rewards = utils.setup_reward_statistics(
            self._reward_positions, reward_statistics
        )

        self._total_rewards = len(self._rewards)
        self._inaccessible_rewards = []

        if not self._keys_change_color and not self._correct_keys_change:
            # here reward_by = color and reward_by = position are equivalent
            pass

        if self._correct_keys_change:
            pass
        else:
            if self._correct_keys is None:
                self._correct_keys = utils.random_correct_keys(
                    num_keys=self._total_rewards
                )
            if self._reward_by == constants.COLOR:
                if self._keys_change_color:
                    # this should never be since then correct_keys_change should be True
                    pass
                else:
                    if (
                        silver_key_positions is not None
                        and gold_key_positions is not None
                    ):
                        self._silver_key_positions = silver_key_positions
                        self._gold_key_positions = gold_key_positions
                        self._default_color_indices()

                    else:
                        self._color_keys_randomly()
            elif self._reward_by == constants.POSITION:
                if not self._keys_change_color:
                    if (
                        silver_key_positions is not None
                        and gold_key_positions is not None
                    ):
                        self._silver_key_positions = silver_key_positions
                        self._gold_key_positions = gold_key_positions
                        self._default_color_indices()

                    else:
                        self._color_keys_randomly()
                else:
                    pass
            else:
                raise ValueError(
                    f"self._reward_by {self._reward_by} is not recognised."
                )

        (
            self._positional_state_space,
            self._key_1_possession_state_space,
            self._key_2_possession_state_space,
            self._rewards_received_state_space,
            self._state_space,
            self._wall_state_space,
        ) = utils.configure_posner_state_space(
            map_outline=self._map,
            key_1_positions=self._key_1_positions,
            key_2_positions=self._key_2_positions,
            reward_positions=self._reward_positions,
        )

    def _default_color_indices(self):
        self._silver_key_indices = []
        self._gold_key_indices = []

        self._silver_key_1s, self._silver_key_2s = [], []
        self._gold_key_1s, self._gold_key_2s = [], []

        for i, key in enumerate(self._correct_keys):
            if key == 0:
                if self._key_1_positions[i] == self._silver_key_positions[i]:
                    self._silver_key_indices.append(i)
                else:
                    self._gold_key_indices.append(i)
            else:
                if self._key_2_positions[i] == self._silver_key_positions[i]:
                    self._silver_key_indices.append(i)
                else:
                    self._gold_key_indices.append(i)

            if self._key_1_positions[i] == self._silver_key_positions[i]:
                self._silver_key_1s.append(i)
                self._gold_key_2s.append(i)
            else:
                self._silver_key_2s.append(i)
                self._gold_key_1s.append(i)

    def _color_keys_randomly(self) -> None:
        """Assigns two disjoint lists of equal length such that the set of the two equals
        the set of key_1_positions and key_2_positions. Allocation is random."""
        self._silver_key_positions, self._gold_key_positions = [], []
        self._silver_key_indices, self._gold_key_indices = [], []
        self._silver_key_1s, self._silver_key_2s = [], []
        self._gold_key_1s, self._gold_key_2s = [], []
        indices = np.random.random(size=self._total_rewards) < 0.5

        for i, (key_1, key_2) in enumerate(
            zip(self._key_1_positions.values(), self._key_2_positions.values())
        ):
            if indices[i]:
                # key 1 is silver, key 2 is gold
                self._silver_key_positions.append(key_1)
                self._gold_key_positions.append(key_2)
                self._silver_key_1s.append(i)
                self._gold_key_2s.append(i)
                if self._correct_keys[i] == 0:
                    self._silver_key_indices.append(i)
                else:
                    self._gold_key_indices.append(i)
            else:
                # key 1 is gold, key 2 is silver
                self._silver_key_positions.append(key_2)
                self._gold_key_positions.append(key_1)
                self._silver_key_2s.append(i)
                self._gold_key_1s.append(i)
                if self._correct_keys[i] == 1:
                    self._silver_key_indices.append(i)
                else:
                    self._gold_key_indices.append(i)

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
                    self._key_1_possession_state_space,
                    self._key_2_possession_state_space,
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
        for key_1_state in self._key_1_possession_state_space:
            for key_2_state in self._key_2_possession_state_space:
                for reward_state in self._rewards_received_state_space:
                    value_combination = {}
                    for state in self._positional_state_space:
                        value_combination[state] = values[
                            state + key_1_state + key_2_state + reward_state
                        ]
                    value_combinations[
                        key_1_state + key_2_state + reward_state
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
        skeleton = np.ones(self._skeleton_shape)

        # make walls black
        skeleton[self._map == 1] = np.zeros(3)

        if rewards is not None:
            if isinstance(rewards, str):
                if rewards == constants.STATIONARY:
                    reward_iterate = self._reward_positions
                elif rewards == constants.STATE:

                    # wrong_keys_collected = []
                    # for i, k in enumerate(self._correct_keys):
                    #     if k == 0 and self._keys_2_state[i]:
                    #         wrong_keys_collected.append(True)
                    #     elif k == 1 and self._keys_1_state[i]:
                    #         wrong_keys_collected.append(True)
                    #     else:
                    #         wrong_keys_collected.append(False)

                    wrong_keys_1 = self._keys_1_state * self._correct_keys
                    wrong_keys_2 = (
                        self._keys_2_state - self._correct_keys == 1
                    ).astype(int)
                    wrong_keys_collected = wrong_keys_1 + wrong_keys_2

                    reward_iterate = [
                        self._reward_positions[i]
                        for i, r in enumerate(self._rewards_state)
                        if not r and not wrong_keys_collected[i]
                    ]
                else:
                    raise ValueError(f"Rewards keyword {rewards} not identified.")
            elif isinstance(rewards, tuple):
                reward_iterate = [
                    self._reward_positions[i] for i, r in enumerate(rewards) if not r
                ]
            else:
                raise ValueError(
                    "Rewards must be string with relevant keyword or keystate list"
                )
            # show reward in green
            for reward in reward_iterate:
                skeleton[reward[::-1]] = self.GREEN_RGB

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
                skeleton[tuple(key_position[::-1])] = self.SILVER_RGB

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
                skeleton[tuple(key_position[::-1])] = self.GOLD_RGB

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
                skeleton[tuple(door[::-1])] = self.RED_RGB

        if agent is not None:
            if isinstance(agent, str):
                if agent == constants.STATE:
                    agent_position = self._agent_position
                elif agent == constants.STATIONARY:
                    agent_position = self._starting_xy
            else:
                agent_position = agent
            # show agent in blue
            skeleton[tuple(agent_position[::-1])] = self.BLUE_RGB

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

    def _rolling_cued_skeleton(self):
        return np.vstack((self._current_cue, self._rolling_env_skeleton))

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
                + tuple(self._keys_1_state)
                + tuple(self._keys_2_state)
                + tuple(self._rewards_state)
            )
        elif self._representation in [constants.PIXEL, constants.PO_PIXEL]:
            if tuple_state is None:
                state = self._rolling_cued_skeleton()  # H x W x C
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
        provisional_old_position = copy.deepcopy(self._agent_position)
        provisional_new_position = self._agent_position + delta

        moving_into_wall = tuple(provisional_new_position) in self._wall_state_space
        moving_into_door = tuple(provisional_new_position) in self._door_positions
        moving_from_door = tuple(provisional_old_position) in self._door_positions

        if moving_into_door:
            door_index = self._door_positions.index(tuple(provisional_new_position))
            if self._keys_1_state[door_index] or self._keys_2_state[door_index]:
                moving_into_door = False

        if not moving_into_wall and not moving_into_door:
            self._agent_position = provisional_new_position
            # change color of relevant squares in rolling_env_skeleton
            self._rolling_env_skeleton[
                tuple(self._agent_position[::-1])
            ] = self.BLUE_RGB
            if moving_from_door:
                self._rolling_env_skeleton[
                    tuple(provisional_old_position[::-1])
                ] = self.RED_RGB
            else:
                self._rolling_env_skeleton[
                    tuple(provisional_old_position[::-1])
                ] = self.WHITE_RGB

        key_1_index = self._key_1_position_index_mapping.get(
            tuple(self._agent_position)
        )
        key_2_index = self._key_2_position_index_mapping.get(
            tuple(self._agent_position)
        )

        silver_key_collected = False
        gold_key_collected = False

        if key_1_index is not None:
            key_index = key_1_index
            # if neither key has been collected yet in this room
            if (
                not self._keys_1_state[key_1_index]
                and not self._keys_2_state[key_1_index]
            ):
                # change key 1 state to indicate collection
                self._keys_1_state[key_1_index] = 1
                if self._cue_format is not None:
                    # update cue
                    self._current_cue = next(self._cues)
                if key_1_index in self._silver_key_1s:
                    # if key is silver
                    silver_key_collected = True
                else:
                    # if key is gold
                    gold_key_collected = True
                if self._correct_keys[key_1_index] == 1:
                    # wrong key collected
                    next_reward_pos = self._reward_positions[key_1_index]
                    self._inaccessible_rewards.append(next_reward_pos)
                    self._rolling_env_skeleton[next_reward_pos[::-1]] = self.WHITE_RGB

        if key_2_index is not None:
            key_index = key_2_index
            # if neither key has been collected yet in this room
            if (
                not self._keys_2_state[key_2_index]
                and not self._keys_1_state[key_2_index]
            ):
                # change key 2 state to indicate collection
                self._keys_2_state[key_2_index] = 1
                if self._cue_format is not None:
                    # update cue
                    self._current_cue = next(self._cues)
                if key_2_index in self._silver_key_2s:
                    # if key is silver
                    silver_key_collected = True
                else:
                    # if key is gold
                    gold_key_collected = True
                if self._correct_keys[key_2_index] == 0:
                    # wrong key collected
                    next_reward_pos = self._reward_positions[key_2_index]
                    self._inaccessible_rewards.append(next_reward_pos)
                    self._rolling_env_skeleton[next_reward_pos[::-1]] = self.WHITE_RGB

        if silver_key_collected:
            self._silver_keys_state[key_index] = 1
            # silver key collected, remove gold key from rolling_env_skeleton
            gold_key_pos = self._gold_key_positions[key_index]
            self._rolling_env_skeleton[tuple(gold_key_pos[::-1])] = self.WHITE_RGB
        elif gold_key_collected:
            self._gold_keys_state[key_index] = 1
            # gold key collected, remove silver key from rolling_env_skeleton
            silver_key_pos = self._silver_key_positions[key_index]
            self._rolling_env_skeleton[tuple(silver_key_pos[::-1])] = self.WHITE_RGB

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

        self._active = self._remain_active(reward=reward)
        self._episode_step_count += 1

        full_skeleton = self._rolling_cued_skeleton()

        if self._training:
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            self._train_episode_position_history.append(tuple(self._agent_position))
            self._train_episode_history.append(full_skeleton)
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history.append(
                    self._partial_observation(
                        state=full_skeleton,
                        agent_position=self._agent_position,
                    )
                )
        else:
            self._test_episode_position_history.append(tuple(self._agent_position))
            self._test_episode_history.append(full_skeleton)
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history.append(
                    self._partial_observation(
                        state=full_skeleton,
                        agent_position=self._agent_position,
                    )
                )

        return reward, new_state

    def _compute_reward(self) -> float:
        """Check for reward, i.e. whether agent position is equal to a reward position.
        If reward is found, add to rewards received log.
        """
        reward = 0.0
        if (
            tuple(self._agent_position) in self._rewards
            and tuple(self._agent_position) not in self._rewards_received
            and tuple(self._agent_position) not in self._inaccessible_rewards
        ):
            reward += self._rewards.get(tuple(self._agent_position))()
            reward_index = self._reward_positions.index(tuple(self._agent_position))
            self._rewards_state[reward_index] = 1
            self._rewards_received.append(tuple(self._agent_position))

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
            len(self._rewards_received)
            == self._total_rewards - len(self._inaccessible_rewards),
        ]
        return not any(conditions)

    def _setup_posner_cues(self):
        cues = []
        # P(cue = gold | key = gold) = P(key = gold | cue = gold)
        # if prior is uniform, which is the case if we have a single cue validity
        cue_conditional = self._cue_validity
        # else via Bayes theorem it more generally would be:
        # cue_conditional = (
        # 2 * silver_cue_validity * gold_cue_validity - gold_cue_validity
        # ) / (silver_cue_validity + gold_cue_validity - 1)

        for i, _ in enumerate(self._correct_keys):
            cue_line = np.tile(
                self.BLACK_RGB, [self._cue_line_depth, self._map.shape[1], 1]
            )
            distractor_directions = np.repeat(
                np.random.random(self._num_cues) > 0.5, self._cue_size
            )

            cue_line[
                :, self._cue_size * np.where(distractor_directions), :
            ] = self.SILVER_RGB
            cue_line[
                :, self._cue_size * np.where(distractor_directions == False), :
            ] = self.GOLD_RGB

            random_boolean = np.random.random() < cue_conditional

            if i in self._silver_key_indices:
                if random_boolean:
                    cue_pixel = self.SILVER_RGB
                else:
                    cue_pixel = self.GOLD_RGB
            else:
                if random_boolean:
                    cue_pixel = self.GOLD_RGB
                else:
                    cue_pixel = self.SILVER_RGB

            cue_line[
                :,
                self._cue_size
                * self._cue_index : self._cue_size
                * (self._cue_index + 1),
                :,
            ] = cue_pixel

            cues.append(cue_line)

        cues.append(self._default_cue_line)

        return iter(cues)

    def _setup_block_cues(self):
        cues = []
        # P(cue = gold | key = gold) = P(key = gold | cue = gold)
        # if prior is uniform, which is the case if we have a single cue validity
        cue_conditional = self._cue_validity
        # else via Bayes theorem it more generally would be:
        # cue_conditional = (
        # 2 * silver_cue_validity * gold_cue_validity - gold_cue_validity
        # ) / (silver_cue_validity + gold_cue_validity - 1)

        for i, _ in enumerate(self._correct_keys):

            random_boolean = np.random.random() < cue_conditional

            if i in self._silver_key_indices:
                if random_boolean:
                    pixel = self.SILVER_RGB
                else:
                    pixel = self.GOLD_RGB
            else:
                if random_boolean:
                    pixel = self.GOLD_RGB
                else:
                    pixel = self.SILVER_RGB

            cue_line = np.tile(
                pixel, [self._cue_line_depth, self._skeleton_shape[1], 1]
            )
            cues.append(cue_line)

        cues.append(self._default_cue_line)

        return iter(cues)

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
        self._inaccessible_rewards = []
        self._keys_1_state = np.zeros(len(self._key_1_positions), dtype=int)
        self._keys_2_state = np.zeros(len(self._key_2_positions), dtype=int)
        self._rewards_state = np.zeros(len(self._rewards), dtype=int)

        if self._correct_keys_change:
            self._correct_keys = utils.random_correct_keys(num_keys=self._total_rewards)
            if self._reward_by == constants.COLOR:
                if self._keys_change_color:
                    self._color_keys_randomly()
                else:
                    raise ValueError("Inconsistent configuration.")
            elif self._reward_by == constants.POSITION:
                if self._keys_change_color:
                    self._color_keys_randomly()
                else:
                    pass
            else:
                raise ValueError(
                    f"self._reward_by {self._reward_by} is not recognised."
                )
        else:
            if self._reward_by == constants.COLOR:
                if self._keys_change_color:
                    # this should never be since then correct_keys_change should be True
                    raise ValueError("Inconsistent configuration.")
                else:
                    pass
            elif self._reward_by == constants.POSITION:
                if not self._keys_change_color:
                    pass
                else:
                    raise NotImplementedError
            else:
                raise ValueError(
                    f"self._reward_by {self._reward_by} is not recognised."
                )

        self._silver_keys_state = np.zeros(len(self._silver_key_positions), dtype=int)
        self._gold_keys_state = np.zeros(len(self._gold_key_positions), dtype=int)

        if self._cue_format == constants.POSNER:
            self._cues = self._setup_posner_cues()
            self._current_cue = next(self._cues)
        elif self._cue_format == constants.SINGLE_BAR:
            self._cues = self._setup_block_cues()
            self._current_cue = next(self._cues)
        else:
            self._current_cue = self._default_cue_line

        skeleton = self._env_skeleton()
        full_skeleton = np.vstack((self._current_cue, skeleton))

        if train:
            self._train_episode_position_history = [tuple(self._agent_position)]
            self._train_episode_history = [full_skeleton]
            self._visitation_counts[self._agent_position[1]][
                self._agent_position[0]
            ] += 1
            if self._representation == constants.PO_PIXEL:
                self._train_episode_partial_history = [
                    self._partial_observation(
                        state=full_skeleton, agent_position=self._agent_position
                    )
                ]
        else:
            self._test_episode_position_history = [tuple(self._agent_position)]
            self._test_episode_history = [full_skeleton]
            if self._representation == constants.PO_PIXEL:
                self._test_episode_partial_history = [
                    self._partial_observation(
                        state=full_skeleton, agent_position=self._agent_position
                    )
                ]

        self._rolling_env_skeleton = skeleton

        initial_state = self.get_state_representation()

        return initial_state
