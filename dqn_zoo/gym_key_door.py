import os
import shutil
from typing import List, Optional, Tuple, Union

import dm_env
import numpy as np
from dm_env import specs

from dqn_zoo.key_door import curriculum_env, posner_env, visualisation_env


class GymKeyDoor(dm_env.Environment):
    """Gym KeyDoor with a `dm_env.Environment` interface."""

    def __init__(
        self,
        env_args,
        train_index,
        test_index,
        env_shape,
        checkpoint_path,
        curriculum_args=None,
    ):
        self._env_args = env_args
        self._curriculum_args = curriculum_args
        self._env_shape = env_shape
        self._checkpoint_path = checkpoint_path
        self._map_path = os.path.join(self._checkpoint_path, "maps")
        self._rollout_path = os.path.join(self._checkpoint_path, "rollouts")
        os.makedirs(self._rollout_path, exist_ok=True)
        os.makedirs(self._map_path, exist_ok=True)

        self._start_of_episode = True
        self._train_index = train_index
        self._test_index = test_index
        self._training: bool = False

        self._key_door_env = posner_env.PosnerEnv(**env_args)
        self._key_door_env = visualisation_env.VisualisationEnv(self._key_door_env)
        if curriculum_args is not None and curriculum_args["apply"]:
            self._curriculum = True
            self._key_door_env = curriculum_env.CurriculumEnv(
                self._key_door_env,
                transitions=curriculum_args["map_yaml_paths"],
            )
            if curriculum_args["transition_episodes"] is None:
                self._next_transition_episode = np.inf
            else:
                self._transition_episodes = iter(curriculum_args["transition_episodes"])
                self._next_transition_episode = next(self._transition_episodes)
        else:
            self._curriculum = False

        self._save_env_data()
        self._save_environment_images()

    def _save_env_data(self):
        # save map source files to exp_path
        shutil.copy(self._env_args["map_ascii_path"], self._checkpoint_path)
        shutil.copy(self._env_args["map_yaml_path"], self._checkpoint_path)
        if self._curriculum:
            curriculum_spec_path = os.path.join(
                self._checkpoint_path, "curriculum_spec"
            )
            os.makedirs(curriculum_spec_path, exist_ok=True)
            for i, yaml_path in enumerate(self._curriculum_args["map_yaml_paths"]):
                filename = os.path.splitext(os.path.basename(yaml_path))[0]
                ext = os.path.splitext(yaml_path)[-1]
                shutil.copy(yaml_path, os.path.join(curriculum_spec_path, f"{filename}_{i}{ext}"))
        

    def reset(self, train: bool) -> dm_env.TimeStep:
        """Resets the environment and starts a new episode."""
        if self._training:
            if self._train_index % 100 == 0 and self._train_index != 0:
                try:
                    self._key_door_env.visualise_episode_history(
                        os.path.join(
                            self._rollout_path,
                            f"train_episode_{self._train_index}.mp4",
                        )
                    )
                except:
                    print(self._train_index, self._test_index, "FAILED VISUALISATION")

            if self._curriculum and self._train_index == self._next_transition_episode:
                self._transition_environment()
        else:
            if self._test_index % 10 == 0 and self._test_index != 0:
                try:
                    self._key_door_env.visualise_episode_history(
                        os.path.join(
                            self._rollout_path,
                            f"test_episode_{self._test_index}.mp4",
                        ),
                        history="test",
                    )
                except:
                    print(
                        self._train_index, self._test_index, "FAILED VISUALISATION TEST"
                    )
        self._training = train
        observation = self._key_door_env.reset_environment(train=train)
        lives = np.int32(1)
        timestep = dm_env.restart((observation, lives))
        self._start_of_episode = False
        if train:
            self._train_index += 1
        else:
            self._test_index += 1
        return timestep

    def _transition_environment(self):
        next(self._key_door_env)
        try:
            self._next_transition_episode = next(self._transition_episodes)
        except StopIteration:
            self._next_transition_episode = np.inf
        except AttributeError:
            self._next_transition_episode = np.inf

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
            specs.Array(shape=self._env_shape, dtype=np.float64, name="rgb"),
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

    def get_state_representation(
        self,
        tuple_state: Optional[Tuple] = None,
    ) -> Union[tuple, np.ndarray]:
        return self._key_door_env.get_state_representation(tuple_state=tuple_state)

    def average_values_over_positional_states(self, values):
        return self._key_door_env.average_values_over_positional_states(values)

    def plot_heatmap_over_env(self, heatmap, save_name):
        return self._key_door_env.plot_heatmap_over_env(
            heatmap=heatmap, save_name=save_name
        )

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._key_door_env.state_space

    @property
    def train_index(self):
        return self._train_index

    @property
    def test_index(self):
        return self._test_index
    
    @property
    def state_visitation_counts(self):
        return self._key_door_env.state_visitation_counts

    def _save_environment_images_for_yaml(self, yaml_path, extra=""):
        self._key_door_env.reset_environment(map_yaml_path=yaml_path)
        plain_map_path = os.path.join(self._map_path, f"plain_map{extra}.pdf")
        cue_map_path = os.path.join(self._map_path, f"cue_map_path{extra}.pdf")
        bounding_box_map_path = os.path.join(self._map_path, f"bounding_box_map{extra}.pdf")
        annotated_map_path = os.path.join(self._map_path, f"annotated_map{extra}.pdf")
        # grid without cue
        self._key_door_env.render(save_path=plain_map_path)
        # grid with cue (if applicable)
        self._key_door_env.render(save_path=cue_map_path, format="cue")
        # grid with cue and cue index highlighted
        self._key_door_env.render(
            save_path=bounding_box_map_path, format="cue", annotate=True
        )
        # grid with cue, cue index highlighted, validity annotated
        self._key_door_env.render(
            save_path=annotated_map_path, format="cue", annotate="full"
        )

    def _save_environment_images(self):
        self._save_environment_images_for_yaml(yaml_path=None)
        if self._curriculum:
            for i, yaml_path in enumerate(self._curriculum_args["map_yaml_paths"]):
                self._save_environment_images_for_yaml(yaml_path=yaml_path, extra=f"_{i + 1}")


class RandomNoopsEnvironmentWrapper(dm_env.Environment):
    """Adds a random number of noop actions at the beginning of each episode."""

    def __init__(
        self,
        environment: dm_env.Environment,
        max_noop_steps: int,
        min_noop_steps: int = 0,
        noop_action: int = None,
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

    def reset(self, train: bool):
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
        return self._apply_random_noops(
            initial_timestep=self._environment.reset(train=train)
        )

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

    def get_state_representation(
        self,
        tuple_state: Optional[Tuple] = None,
    ) -> Union[tuple, np.ndarray]:
        return self._environment.get_state_representation(tuple_state=tuple_state)

    def average_values_over_positional_states(self, values):
        return self._environment.average_values_over_positional_states(values)

    def plot_heatmap_over_env(self, heatmap, save_name):
        return self._environment.plot_heatmap_over_env(heatmap, save_name)
    
    def transition_environment(self):
        self._environment._transition_environment()

    @property
    def state_space(self) -> List[Tuple[int, int]]:
        return self._environment.state_space

    @property
    def train_index(self):
        return self._environment.train_index

    @property
    def test_index(self):
        return self._environment.test_index
    
    @property
    def state_visitation_counts(self):
        return self._environment.state_visitation_counts
        