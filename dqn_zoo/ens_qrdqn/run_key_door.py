# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A QR-DQN agent training on Atari.

From the paper "Distributional Reinforcement Learning with Quantile Regression"
http://arxiv.org/abs/1710.10044.
"""

# pylint: disable=g-bad-import-order

import collections
import datetime
import itertools
import json
import os
import shutil
import sys
import time
import typing

import chex
import dm_env
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import app, flags, logging
from jax.config import config

from dqn_zoo import atari_data, constants, gym_key_door, networks, parts, processors
from dqn_zoo import replay as replay_lib
from dqn_zoo.ens_qrdqn import agent

# Relevant flag values are expressed in terms of environment frames.
FLAGS = flags.FLAGS
flags.DEFINE_string("environment_name", "posner_env", "")
flags.DEFINE_integer("environment_height", 84, "")
flags.DEFINE_integer("environment_width", 84, "")
flags.DEFINE_integer("replay_capacity", int(1e6), "")
flags.DEFINE_bool("compress_state", True, "")
flags.DEFINE_bool("grayscale", False, "")
flags.DEFINE_float("min_replay_capacity_fraction", 0.05, "")
flags.DEFINE_integer("batch_size", 32, "")
flags.DEFINE_integer("max_frames_per_episode", 800, "")  # 30 mins.
flags.DEFINE_integer("num_action_repeats", 1, "")
flags.DEFINE_integer("num_stacked_frames", 4, "")
flags.DEFINE_float("exploration_epsilon_begin_value", 1.0, "")
flags.DEFINE_float("exploration_epsilon_end_value", 0.1, "")
flags.DEFINE_float("exploration_epsilon_decay_frame_fraction", 0.02, "")
flags.DEFINE_float("eval_exploration_epsilon", 0.05, "")
flags.DEFINE_integer("target_network_update_period", int(1e4), "")
flags.DEFINE_float("huber_param", 1.0, "")
flags.DEFINE_float("learning_rate", 0.00025, "")
flags.DEFINE_float("optimizer_epsilon", 0.01 / 32**2, "")
flags.DEFINE_float("additional_discount", 0.99, "")
flags.DEFINE_float("max_abs_reward", 1.0, "")
flags.DEFINE_float("max_global_grad_norm", 10.0, "")
flags.DEFINE_integer("seed", 1, "")  # GPU may introduce nondeterminism.
flags.DEFINE_integer("num_iterations", 1500, "")
flags.DEFINE_integer("num_train_frames", int(1e4), "")  # Per iteration.
flags.DEFINE_integer("num_eval_frames", int(1e4), "")  # Per iteration.
flags.DEFINE_integer("learn_period", 4, "")
# flags.DEFINE_string("results_csv_path", "/tmp/results.csv", "")
flags.DEFINE_string("results_path", None, "")  # where to store results

flags.DEFINE_string("map_ascii_path", "dqn_zoo/key_door_maps/multi_room_bandit.txt", "")
flags.DEFINE_string("map_yaml_path", "dqn_zoo/key_door_maps/multi_room_bandit.yaml", "")
flags.DEFINE_bool("apply_curriculum", True, "")
flags.DEFINE_list(
    "map_yaml_paths",
    [
        "dqn_zoo/key_door_maps/multi_room_bandit.yaml",
        "dqn_zoo/key_door_maps/multi_room_bandit_1.yaml",
        "dqn_zoo/key_door_maps/multi_room_bandit_2.yaml",
    ],
    "Set of yaml paths that define different contexts. Used only if apply_curriculum is True",
)
flags.DEFINE_multi_integer(
    "transition_iterations",
    (500, 1000),
    "Iteration number at which environment context switches. Should have same dimension as map_yaml_paths",
)
flags.DEFINE_integer("env_scaling", 8, "")
flags.DEFINE_multi_integer("env_shape", (84, 84, 12), "")

flags.DEFINE_integer("num_quantiles", 201, "")
flags.DEFINE_integer("ens_size", 8, "")
flags.DEFINE_float("mask_probability", 1.0, "")

flags.DEFINE_bool("prioritise", None, "")
flags.DEFINE_float("priority_exponent", 0.6, "")
flags.DEFINE_float("importance_sampling_exponent_begin_value", 0.4, "")
flags.DEFINE_float("importance_sampling_exponent_end_value", 1.0, "")
flags.DEFINE_float("uniform_sample_probability", 1e-3, "")
flags.DEFINE_bool("normalize_weights", True, "")


def main(argv):
    """Trains QR-DQN Ensemble agent on Key-Door."""
    del argv
    logging.info(
        "QR-DQN Ensemble on Key-Door on %s.", jax.lib.xla_bridge.get_backend().platform
    )
    random_state = np.random.RandomState(FLAGS.seed)
    rng_key = jax.random.PRNGKey(
        random_state.randint(-sys.maxsize - 1, sys.maxsize + 1, dtype=np.int64)
    )

    # create timestamp for logging and checkpoint path
    if FLAGS.results_path is None:
        raw_datetime = datetime.datetime.fromtimestamp(time.time())
        exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        exp_path = os.path.join("results", exp_timestamp)
        os.makedirs(exp_path, exist_ok=True)
    else:
        exp_path = FLAGS.results_path

    flag_dict = FLAGS.flag_values_dict()
    with open(os.path.join(exp_path, "flags.json"), "+w") as json_file:
        json.dump(flag_dict, json_file, indent=6)

    visualisation_path = os.path.join(exp_path, "visualisations")
    os.makedirs(visualisation_path, exist_ok=True)

    if FLAGS.apply_curriculum:
        transition_iterations = iter(FLAGS.transition_iterations)
        next_transition_iteration = next(transition_iterations)

    def environment_builder(train_index: int = 0, test_index: int = 0):
        """Creates Atari environment."""
        env = gym_key_door.GymKeyDoor(
            env_args={
                constants.MAP_ASCII_PATH: FLAGS.map_ascii_path,
                constants.MAP_YAML_PATH: FLAGS.map_yaml_path,
                constants.REPRESENTATION: constants.PIXEL,
                constants.SCALING: FLAGS.env_scaling,
                constants.EPISODE_TIMEOUT: FLAGS.max_frames_per_episode,
                constants.GRAYSCALE: False,
                constants.BATCH_DIMENSION: False,
                constants.TORCH_AXES: False,
            },
            train_index=train_index,
            test_index=test_index,
            env_shape=FLAGS.env_shape,
            checkpoint_path=exp_path,
            curriculum_args={
                "apply": FLAGS.apply_curriculum,
                "map_yaml_paths": FLAGS.map_yaml_paths,
                "transition_episodes": None,
            },
            # FLAGS.environment_name, seed=random_state.randint(1, 2**32)
        )
        return gym_key_door.RandomNoopsEnvironmentWrapper(
            env,
            min_noop_steps=0,
            max_noop_steps=0,
            seed=random_state.randint(1, 2**32),
        )

    env = environment_builder()

    logging.info("Environment: %s", FLAGS.environment_name)
    logging.info("Action spec: %s", env.action_spec())
    logging.info("Observation spec: %s", env.observation_spec())
    num_actions = env.action_spec().num_values
    num_quantiles = FLAGS.num_quantiles
    quantiles = (jnp.arange(0, num_quantiles) + 0.5) / float(num_quantiles)
    network_fn = networks.ens_qr_atari_network(num_actions, quantiles, FLAGS.ens_size)
    network = hk.transform(network_fn)

    def preprocessor_builder():
        return processors.atari(
            additional_discount=FLAGS.additional_discount,
            max_abs_reward=FLAGS.max_abs_reward,
            resize_shape=(FLAGS.environment_height, FLAGS.environment_width),
            num_action_repeats=FLAGS.num_action_repeats,
            num_pooled_frames=2,
            zero_discount_on_life_loss=True,
            num_stacked_frames=FLAGS.num_stacked_frames,
            grayscaling=FLAGS.grayscale,
        )

    # Create sample network input from sample preprocessor output.
    sample_processed_timestep = preprocessor_builder()(env.reset(train=None))
    sample_processed_timestep = typing.cast(dm_env.TimeStep, sample_processed_timestep)
    sample_network_input = sample_processed_timestep.observation

    if FLAGS.grayscale:
        num_channels = 1
    else:
        num_channels = 3

    chex.assert_shape(
        sample_network_input,
        (
            FLAGS.environment_height,
            FLAGS.environment_width,
            num_channels * FLAGS.num_stacked_frames,
        ),
    )

    exploration_epsilon_schedule = parts.LinearSchedule(
        begin_t=int(
            FLAGS.min_replay_capacity_fraction
            * FLAGS.replay_capacity
            * FLAGS.num_action_repeats
        ),
        decay_steps=int(
            FLAGS.exploration_epsilon_decay_frame_fraction
            * FLAGS.num_iterations
            * FLAGS.num_train_frames
        ),
        begin_value=FLAGS.exploration_epsilon_begin_value,
        end_value=FLAGS.exploration_epsilon_end_value,
    )

    if FLAGS.compress_state:

        def encoder(transition):
            return transition._replace(
                s_tm1=replay_lib.compress_array(transition.s_tm1),
                s_t=replay_lib.compress_array(transition.s_t),
            )

        def decoder(transition):
            return transition._replace(
                s_tm1=replay_lib.uncompress_array(transition.s_tm1),
                s_t=replay_lib.uncompress_array(transition.s_t),
            )

    else:
        encoder = None
        decoder = None

    replay_structure = replay_lib.MaskedTransition(
        s_tm1=None,
        a_tm1=None,
        r_t=None,
        discount_t=None,
        s_t=None,
        mask_t=None,
    )

    if FLAGS.prioritise is not None:
        importance_sampling_exponent_schedule = parts.LinearSchedule(
            begin_t=int(FLAGS.min_replay_capacity_fraction * FLAGS.replay_capacity),
            end_t=(
                FLAGS.num_iterations
                * int(FLAGS.num_train_frames / FLAGS.num_action_repeats)
            ),
            begin_value=FLAGS.importance_sampling_exponent_begin_value,
            end_value=FLAGS.importance_sampling_exponent_end_value,
        )
        replay = replay_lib.PrioritizedTransitionReplay(
            FLAGS.replay_capacity,
            replay_structure,
            FLAGS.priority_exponent,
            importance_sampling_exponent_schedule,
            FLAGS.uniform_sample_probability,
            FLAGS.normalize_weights,
            random_state,
            encoder,
            decoder,
        )
    else:
        replay = replay_lib.TransitionReplay(
            FLAGS.replay_capacity, replay_structure, random_state, encoder, decoder
        )

    optimizer = optax.adam(
        learning_rate=FLAGS.learning_rate, eps=FLAGS.optimizer_epsilon
    )
    if FLAGS.max_global_grad_norm > 0:
        optimizer = optax.chain(
            optax.clip_by_global_norm(FLAGS.max_global_grad_norm), optimizer
        )

    train_rng_key, eval_rng_key = jax.random.split(rng_key)

    # create timestamp for logging and checkpoint path
    if FLAGS.results_path is None:
        raw_datetime = datetime.datetime.fromtimestamp(time.time())
        exp_timestamp = raw_datetime.strftime("%Y-%m-%d-%H-%M-%S")
        exp_path = os.path.join("results", exp_timestamp)
        os.makedirs(exp_path, exist_ok=True)
    else:
        exp_path = FLAGS.results_path

    train_agent = agent.EnsQrDqn(
        preprocessor=preprocessor_builder(),
        sample_network_input=sample_network_input,
        network=network,
        quantiles=quantiles,
        optimizer=optimizer,
        transition_accumulator=replay_lib.TransitionAccumulator(),
        replay=replay,
        prioritise=FLAGS.prioritise,
        ens_size=FLAGS.ens_size,
        mask_probability=FLAGS.mask_probability,
        batch_size=FLAGS.batch_size,
        exploration_epsilon=exploration_epsilon_schedule,
        min_replay_capacity_fraction=FLAGS.min_replay_capacity_fraction,
        learn_period=FLAGS.learn_period,
        target_network_update_period=FLAGS.target_network_update_period,
        huber_param=FLAGS.huber_param,
        rng_key=train_rng_key,
    )
    eval_agent = parts.EpsilonGreedyActor(
        preprocessor=preprocessor_builder(),
        network=network,
        exploration_epsilon=FLAGS.eval_exploration_epsilon,
        rng_key=eval_rng_key,
    )

    # setup writer
    writer = parts.CsvWriter(os.path.join(exp_path, "writer.csv"))

    # setup checkpointing.
    checkpoint = parts.ImplementedCheckpoint(
        checkpoint_path=os.path.join(exp_path, "checkpoint.pkl")
    )

    if checkpoint.can_be_restored():
        checkpoint.restore()
        train_agent.set_state(state=checkpoint.state.train_agent)
        eval_agent.set_state(state=checkpoint.state.eval_agent)
        writer.set_state(state=checkpoint.state.writer)

    state = checkpoint.state
    state.iteration = 0
    state.train_agent = train_agent
    state.eval_agent = eval_agent
    state.random_state = random_state
    state.writer = writer
    if checkpoint.can_be_restored():
        checkpoint.restore()

    while state.iteration <= FLAGS.num_iterations:
        # New environment for each iteration to allow for determinism if preempted.
        if state.iteration == 0:
            env = environment_builder(train_index=0, test_index=0)
        # env = environment_builder()

        if FLAGS.apply_curriculum and state.iteration == next_transition_iteration:
            print(f"Iteration {state.iteration}: Transitioning Environment.")
            env.transition_environment()
            try:
                next_transition_iteration = next(transition_iterations)
            except StopIteration:
                next_transition_iteration = np.inf

        logging.info("Training iteration %d.", state.iteration)
        train_seq = parts.run_loop(train_agent, env, FLAGS.max_frames_per_episode)
        num_train_frames = 0 if state.iteration == 0 else FLAGS.num_train_frames
        train_seq_truncated = itertools.islice(train_seq, num_train_frames)
        train_trackers = parts.make_default_trackers(train_agent)
        train_stats = parts.generate_statistics(train_trackers, train_seq_truncated)

        logging.info("Evaluation iteration %d.", state.iteration)
        eval_agent.network_params = train_agent.online_params
        eval_seq = parts.run_loop(eval_agent, env, FLAGS.max_frames_per_episode)
        eval_seq_truncated = itertools.islice(eval_seq, FLAGS.num_eval_frames)
        eval_trackers = parts.make_default_trackers(eval_agent)
        eval_stats = parts.generate_statistics(eval_trackers, eval_seq_truncated)

        # Logging and checkpointing.
        human_normalized_score = atari_data.get_human_normalized_score(
            FLAGS.environment_name, eval_stats["episode_return"]
        )
        capped_human_normalized_score = np.amin([1.0, human_normalized_score])

        if train_stats["num_episodes"] == 0:
            train_episode_length = np.nan
        else:
            train_episode_length = FLAGS.num_train_frames / train_stats["num_episodes"]
        if eval_stats["num_episodes"] == 0:
            eval_episode_length = np.nan
        else:
            eval_episode_length = FLAGS.num_train_frames / eval_stats["num_episodes"]
        log_output = [
            ("iteration", state.iteration, "%3d"),
            ("frame", state.iteration * FLAGS.num_train_frames, "%5d"),
            ("eval_episode_return", eval_stats["episode_return"], "% 2.2f"),
            ("train_episode_return", train_stats["episode_return"], "% 2.2f"),
            ("eval_num_episodes", eval_stats["num_episodes"], "%3d"),
            ("train_num_episodes", train_stats["num_episodes"], "%3d"),
            ("eval_frame_rate", eval_stats["step_rate"], "%4.0f"),
            ("train_frame_rate", train_stats["step_rate"], "%4.0f"),
            ("train_exploration_epsilon", train_agent.exploration_epsilon, "%.3f"),
            ("train_state_value", train_stats["state_value"], "%.3f"),
            ("normalized_return", human_normalized_score, "%.3f"),
            ("capped_normalized_return", capped_human_normalized_score, "%.3f"),
            ("human_gap", 1.0 - capped_human_normalized_score, "%.3f"),
            ("train_loss", train_stats["_loss"], "%.5f"),
            ("eval_loss", eval_stats["_loss"], "%.5f"),
            (
                "train_episode_length",
                train_episode_length,
                "%.2f",
            ),
            (
                "eval_episode_length",
                eval_episode_length,
                "%.2f",
            ),
            ("td_errors", train_stats.get("td_errors", np.nan), "%.3f"),
            ("mean_q", train_stats.get("mean_q", np.nan), "%.3f"),
            ("mean_q_var", train_stats.get("mean_q_var", np.nan), "%.3f"),
            ("mean_epistemic", train_stats.get("mean_epistemic", np.nan), "%.3f"),
            ("mean_aleatoric", train_stats.get("mean_aleatoric", np.nan), "%.3f"),
            (
                "q_select",
                train_stats.get("q_select", np.nan),
                "%.3f",
            ),
            ("q_var_select", train_stats.get("q_var_select", np.nan), "%.3f"),
            ("epistemic_select", train_stats.get("epistemic_select", np.nan), "%.3f"),
            ("aleatoric_select", train_stats.get("aleatoric_select", np.nan), "%.3f"),
        ]
        log_output_str = ", ".join(("%s: " + f) % (n, v) for n, v, f in log_output)
        logging.info(log_output_str)
        writer.write(collections.OrderedDict((n, v) for n, v, _ in log_output))
        state.iteration += 1
        # checkpoint.save()

    writer.close()


if __name__ == "__main__":
    config.update("jax_platform_name", "cpu")  # Default to GPU.
    config.update("jax_numpy_rank_promotion", "raise")
    config.config_with_absl()
    app.run(main)
