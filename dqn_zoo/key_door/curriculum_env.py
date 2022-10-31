from typing import List

from dqn_zoo.key_door import wrapper


class CurriculumEnv(wrapper.Wrapper):
    """Environment curriculum wrapper to change the environment spec."""

    def __init__(self, env, transitions: List[str]) -> None:
        """Class constructor.

        Args:
            env: environment to wrap.
            transitions: List of paths to yaml confguration files.
        """
        super().__init__(env=env)

        self._transitions = transitions

    def _transition(self) -> None:
        """Transition environment to next phase."""
        next_yaml_path = self._transitions.pop(0)
        self._env.reset_environment(map_yaml_path=next_yaml_path)

    def __next__(self) -> None:
        """Initiate change in environment."""
        if self._transitions:
            self._transition()
        else:
            raise StopIteration("No more transitions specified.")
