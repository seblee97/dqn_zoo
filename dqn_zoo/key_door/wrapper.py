from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from dqn_zoo.key_door import base_environment


class Wrapper(base_environment.BaseEnvironment):
    def __init__(self, env):
        self._env = env

    def step(self, action) -> Tuple[float, Tuple[int, int]]:
        return self._env.step(action)

    def reset_environment(
        self, train: bool = True, map_yaml_path: Optional[str] = None
    ) -> Tuple[int, int, int]:
        return self._env.reset_environment(train=train, map_yaml_path=map_yaml_path)

    @property
    def active(self) -> bool:
        return self._env.active

    @property
    def episode_step_count(self) -> int:
        return self._env.episode_step_count

    @property
    def agent_position(self) -> Tuple[int, int]:
        return self._env.agent_position

    @property
    def action_space(self) -> List[int]:
        return self._env.ACTION_SPACE

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._env.state_space

    @property
    def positional_state_space(self):
        return self._env.positional_state_space

    @property
    def visitation_counts(self) -> np.ndarray:
        return self._env.visitation_counts

    @property
    def train_episode_history(self) -> List[np.ndarray]:
        return self._env.train_episode_history

    @property
    def test_episode_history(self) -> List[np.ndarray]:
        return self._env.test_episode_history

    @property
    def train_episode_partial_history(self) -> List[np.ndarray]:
        return self._env.train_episode_partial_history

    @property
    def test_episode_partial_history(self) -> List[np.ndarray]:
        return self._env.test_episode_partial_history

    @property
    def train_episode_position_history(self) -> np.ndarray:
        return self._env.train_episode_position_history

    @property
    def test_episode_position_history(self) -> np.ndarray:
        return self._env.test_episode_position_history

    def get_state_representation(self, tuple_state: Optional[Tuple] = None):
        return self._env.get_state_representation(tuple_state=tuple_state)

    # including these in the base wrapper is not ideal. Maybe rethink wrappers...
    def render(
        self,
        save_path: Optional[str] = None,
        dpi: Optional[int] = 60,
        format: str = "state",
    ) -> None:
        self._env.render(save_path=save_path, dpi=dpi, format=format)

    def _env_skeleton(
        self,
        rewards: Union[None, str, Tuple[int]] = "state",
        keys: Union[None, str, Tuple[int]] = "state",
        doors: Union[None, str] = "state",
        agent: Union[None, str, np.ndarray] = "state",
    ):
        self._env._env_skeleton(rewards=rewards, keys=keys, doors=doors, agent=agent)

    def visualise_episode_history(
        self, save_path: str, history: Union[str, List[np.ndarray]] = "train"
    ) -> None:
        self._env.visualise_episode_history(save_path=save_path, history=history)

    def plot_heatmap_over_env(
        self,
        heatmap: Dict[Tuple[int, int], float],
        fig: Optional = None,
        ax: Optional = None,
        save_name: Optional[str] = None,
    ) -> None:
        self._env.plot_heatmap_over_env(
            heatmap=heatmap, fig=fig, ax=ax, save_name=save_name
        )

    def average_values_over_positional_states(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], float]:
        return self._env.average_values_over_positional_states(values=values)

    def get_value_combinations(
        self, values: Dict[Tuple[int], float]
    ) -> Dict[Tuple[int], Dict[Tuple[int], float]]:
        return self._env.get_value_combinations(values=values)

    def __next__(self) -> None:
        next(self._env)
