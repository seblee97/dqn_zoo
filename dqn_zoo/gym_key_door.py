from typing import List, Optional, Tuple, Union

import dm_env
import numpy as np
from dm_env import specs

from dqn_zoo.key_door import posner_env, visualisation_env


class GymKeyDoor(dm_env.Environment):
    """Gym KeyDoor with a `dm_env.Environment` interface."""

    def __init__(self, env_args, env_shape):
        self._env_shape = env_shape
        self._key_door_env = posner_env.PosnerEnv(**env_args)
        self._key_door_env = visualisation_env.VisualisationEnv(self._key_door_env)
        # environment = visualisation_env.VisualisationEnv(environment)
        self._start_of_episode = True

        self._run_index = 0

    def reset(self) -> dm_env.TimeStep:
        """Resets the environment and starts a new episode."""
        if self._run_index % 100 == 0 and self._run_index != 0:
            self._key_door_env.visualise_episode_history(
                f"train_dqnkd_{self._run_index}.mp4"
            )
        observation = self._key_door_env.reset_environment()
        lives = np.int32(1)
        timestep = dm_env.restart((observation, lives))
        self._start_of_episode = False
        self._run_index += 1
        return timestep

    def step(self, action: np.int32) -> dm_env.TimeStep:
        """Updates the environment given an action and returns a timestep."""
        # If the previous timestep was LAST then we call reset() on the Gym
        # environment, otherwise step(). Although Gym environments allow you to step
        # through episode boundaries (similar to dm_env) they emit a warning.
        if self._start_of_episode:
            step_type = dm_env.StepType.FIRST
            observation = self._key_door_env.reset_environment()
            discount = None
            reward = None
            done = False
        else:
            reward, observation = self._key_door_env.step(action)
            done = not self._key_door_env.active
            info = ""
            if done:
                assert "TimeLimit.truncated" not in info, "Should never truncate."
                step_type = dm_env.StepType.LAST
                discount = 0.0
            else:
                step_type = dm_env.StepType.MID
                discount = 1.0

        lives = np.int32(1)
        timestep = dm_env.TimeStep(
            step_type=step_type,
            observation=(observation, lives),
            reward=reward,
            discount=discount,
        )
        self._start_of_episode = done
        return timestep

    def observation_spec(self) -> Tuple[specs.Array, specs.Array]:
        return (
            specs.Array(shape=self._env_shape, dtype=np.float, name="rgb"),
            specs.Array(shape=(), dtype=np.int32, name="lives"),
        )

    def action_spec(self) -> specs.DiscreteArray:
        space = self._key_door_env.action_space
        return specs.DiscreteArray(num_values=len(space), dtype=np.int32, name="action")

    def close(self):
        self._key_door_env.close()

    def plot_array_data(
        self, name: str, data: Union[List[np.ndarray], np.ndarray]
    ) -> None:
        """Plot array data to image file.

        Args:
            name: filename for save.
            data: data to save.
        """
        full_path = name

        if isinstance(data, list):
            animator.animate(
                images=data,
                file_name=full_path,
                plot_origin="lower",
                library="matplotlib_animation",
                file_format=".mp4",
            )
        elif isinstance(data, np.ndarray):
            fig = plt.figure()
            plt.imshow(data, origin="lower")
            plt.colorbar()
            fig.savefig(fname=full_path)
            plt.close(fig)


class RandomNoopsEnvironmentWrapper(dm_env.Environment):
    """Adds a random number of noop actions at the beginning of each episode."""

    def __init__(
        self,
        environment: dm_env.Environment,
        max_noop_steps: int,
        min_noop_steps: int = 0,
        noop_action: int = 0,
        seed: Optional[int] = None,
    ):
        """Initializes the random noops environment wrapper."""
        self._environment = environment
        if max_noop_steps < min_noop_steps:
            raise ValueError("max_noop_steps must be greater or equal min_noop_steps")
        self._min_noop_steps = min_noop_steps
        self._max_noop_steps = max_noop_steps
        self._noop_action = noop_action
        self._rng = np.random.RandomState(seed)

    def reset(self):
        """Begins new episode.

        This method resets the wrapped environment and applies a random number
        of noop actions before returning the last resulting observation
        as the first episode timestep. Intermediate timesteps emitted by the inner
        environment (including all rewards and discounts) are discarded.

        Returns:
          First episode timestep corresponding to the timestep after a random number
          of noop actions are applied to the inner environment.

        Raises:
          RuntimeError: if an episode end occurs while the inner environment
            is being stepped through with the noop action.
        """
        return self._apply_random_noops(initial_timestep=self._environment.reset())

    def step(self, action):
        """Steps environment given action.

        If beginning a new episode then random noops are applied as in `reset()`.

        Args:
          action: action to pass to environment conforming to action spec.

        Returns:
          `Timestep` from the inner environment unless beginning a new episode, in
          which case this is the timestep after a random number of noop actions
          are applied to the inner environment.
        """
        if isinstance(action, tuple):
            action = action[0]
        timestep = self._environment.step(action)
        if timestep.first():
            return self._apply_random_noops(initial_timestep=timestep)
        else:
            return timestep

    def _apply_random_noops(self, initial_timestep):
        assert initial_timestep.first()
        num_steps = self._rng.randint(self._min_noop_steps, self._max_noop_steps + 1)
        timestep = initial_timestep
        for _ in range(num_steps):
            timestep = self._environment.step(self._noop_action)
            if timestep.last():
                raise RuntimeError(
                    "Episode ended while applying %s noop actions." % num_steps
                )

        # We make sure to return a FIRST timestep, i.e. discard rewards & discounts.
        return dm_env.restart(timestep.observation)

    ## All methods except for reset and step redirect to the underlying env.

    def observation_spec(self):
        return self._environment.observation_spec()

    def action_spec(self):
        return self._environment.action_spec()

    def reward_spec(self):
        return self._environment.reward_spec()

    def discount_spec(self):
        return self._environment.discount_spec()

    def close(self):
        return self._environment.close()
