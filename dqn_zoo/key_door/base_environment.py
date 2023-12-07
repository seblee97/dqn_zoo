import abc
from typing import List, Optional, Tuple, Union

import numpy as np

from dqn_zoo.key_door import constants


class BaseEnvironment(abc.ABC):
    """Base class for RL environments in key-door framework.

    Abstract methods:
        step: takes action produces reward and next state.
        reset_environment: reset environment and return initial state.
    """

    ACTION_SPACE = [0, 1, 2, 3]
    # 0: LEFT
    # 1: UP
    # 2: RIGHT
    # 3: DOWN

    DELTAS = {
        0: np.array([-1, 0]),
        1: np.array([0, 1]),
        2: np.array([1, 0]),
        3: np.array([0, -1]),
    }

    MAPPING = {
        constants.WALL_CHARACTER: 1,
        constants.START_CHARACTER: 0,
        constants.DOOR_CHARACTER: 0,
        constants.OPEN_CHARACTER: 0,
        constants.KEY_CHARACTER: 0,
        constants.REWARD_CHARACTER: 0,
    }

    def __init__(
        self,
        representation: str,
        episode_timeout: Optional[Union[int, None]] = None,
        frame_stack: Optional[Union[int, None]] = None,
        scaling: Optional[int] = 1,
        field_x: Optional[int] = 1,
        field_y: Optional[int] = 1,
        binarize: bool = False,
        grayscale: bool = True,
        batch_dimension: bool = True,
        torch_axes: bool = True,
    ):
        self._active: bool = False

        self._training: bool
        self._episode_step_count: int
        self._representation = representation

        self._agent_position: np.ndarray

        self._train_episode_position_history: List[List[int]]
        self._test_episode_position_history: List[List[int]]
        self._train_episode_history: List[np.ndarray]
        self._test_episode_history: List[np.ndarray]

        if self._representation == constants.PO_PIXEL:
            self._train_episode_partial_history: List[np.ndarray]
            self._test_episode_partial_history: List[np.ndarray]

        self._episode_timeout = episode_timeout or np.inf
        self._frame_stack = frame_stack or 1
        self._scaling = scaling
        self._field_x = field_x
        self._field_y = field_y
        self._binarize = binarize
        self._grayscale = grayscale
        self._batch_dimension = batch_dimension
        self._torch_axes = torch_axes

    @abc.abstractmethod
    def step(self, action: int) -> Tuple[float, Tuple[int, int]]:
        """Take step in environment according to action of agent."""
        pass

    @abc.abstractmethod
    def reset_environment(self, train: bool):
        """Reset environment.

        Args:
            train: whether episode is for train or test
            (may affect e.g. logging).
        """
        pass

    @property
    def active(self) -> bool:
        return self._active

    @property
    def episode_step_count(self) -> int:
        return self._episode_step_count

    @property
    def agent_position(self) -> Tuple[int, int]:
        return tuple(self._agent_position)

    @property
    def action_space(self) -> List[int]:
        return self.ACTION_SPACE

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._state_space

    @property
    def positional_state_space(self):
        return self._positional_state_space

    @property
    def visitation_counts(self) -> np.ndarray:
        return self._visitation_counts
    
    @property
    def state_visitation_counts(self) -> np.ndarray:
        return self._state_visitation_counts

    @property
    def train_episode_history(self) -> List[np.ndarray]:
        return self._train_episode_history

    @property
    def test_episode_history(self) -> List[np.ndarray]:
        return self._test_episode_history

    @property
    def train_episode_partial_history(self) -> List[np.ndarray]:
        return self._train_episode_partial_history

    @property
    def test_episode_partial_history(self) -> List[np.ndarray]:
        return self._test_episode_partial_history

    @property
    def train_episode_position_history(self) -> np.ndarray:
        return np.array(self._train_episode_position_history)

    @property
    def test_episode_position_history(self) -> np.ndarray:
        return np.array(self._test_episode_position_history)
