from typing import Tuple

import dm_env
import numpy as np
from dm_env import specs
from key_door import key_door_env


class GymKeyDoor(dm_env.Environment):
    """Gym KeyDoor with a `dm_env.Environment` interface."""

    def __init__(self, env_args, env_shape):
        self._env_shape = env_shape
        self._key_door_env = key_door_env.KeyDoorGridworld(**env_args)
        # environment = visualisation_env.VisualisationEnv(environment)
        self._start_of_episode = True

    def reset(self) -> dm_env.TimeStep:
        """Resets the environment and starts a new episode."""
        observation = self._key_door_env.reset_environment()
        lives = np.int32(1)
        timestep = dm_env.restart((observation, lives))
        self._start_of_episode = False
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
