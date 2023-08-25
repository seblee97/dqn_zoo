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
"""QR-DQN agent class."""

# pylint: disable=g-bad-import-order

from typing import Any, Callable, Mapping, Text

import chex
import distrax
import dm_env
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rlax
from absl import logging

from dqn_zoo import parts, processors
from dqn_zoo import replay as replay_lib

# Batch variant of quantile_q_learning with fixed tau input across batch.
_batch_quantile_q_learning = jax.vmap(
    rlax.quantile_q_learning, in_axes=(0, None, 0, 0, 0, 0, 0, None)
)
_select_actions = jax.vmap(lambda q, a: q[a])


class EnsQrDqn(parts.Agent):
    """Ensemble of Quantile Regression DQN agent."""

    def __init__(
        self,
        preprocessor: processors.Processor,
        sample_network_input: jnp.ndarray,
        network: parts.Network,
        quantiles: jnp.ndarray,
        optimizer: optax.GradientTransformation,
        transition_accumulator: Any,
        replay: replay_lib.TransitionReplay,
        prioritise: str,
        mask_probability: float,
        ens_size: int,
        batch_size: int,
        exploration_epsilon: Callable[[int], float],
        min_replay_capacity_fraction: float,
        learn_period: int,
        target_network_update_period: int,
        huber_param: float,
        rng_key: parts.PRNGKey,
    ):
        self._preprocessor = preprocessor
        self._replay = replay
        self._prioritise = prioritise
        self._transition_accumulator = transition_accumulator
        self._mask_probabilities = jnp.array([mask_probability, 1 - mask_probability])
        self._ens_size = ens_size
        self._batch_size = batch_size
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
        self._learn_period = learn_period
        self._target_network_update_period = target_network_update_period

        # Initialize network parameters and optimizer.
        self._rng_key, network_rng_key = jax.random.split(rng_key)

        self._online_params = network.init(
            network_rng_key, (sample_network_input[None, ...], False)
        )
        self._target_params = self._online_params
        self._opt_state = optimizer.init(self._online_params)

        # Other agent state: last action, frame count, etc.
        self._action = None
        self._frame_t = -1  # Current frame index.
        self._statistics = {"state_value": np.nan}
        self._max_seen_priority = 1.0

        # Define jitted loss, update, and policy functions here instead of as
        # class methods, to emphasize that these are meant to be pure functions
        # and should not access the agent object's state via `self`.

        def loss_fn(
            online_params, target_params, transitions, weights, rng_key, stop_grad
        ):
            """Calculates loss given network parameters and transitions."""
            # Compute Q value distributions.
            _, online_key, target_key = jax.random.split(rng_key, 3)
            dist_q_tm1 = network.apply(
                online_params, online_key, (transitions.s_tm1, stop_grad)
            )
            dist_q_target_t = network.apply(
                target_params, target_key, (transitions.s_t, False)
            )

            # Batch x Ensemble : Quantiles: Actions
            flattened_dist_q_tm1 = jnp.reshape(
                dist_q_tm1.q_dist,
                (-1, dist_q_tm1.q_dist.shape[2], dist_q_tm1.q_dist.shape[3]),
            )
            flattened_dist_q_target_t = jnp.reshape(
                dist_q_target_t.q_dist,
                (-1, dist_q_target_t.q_dist.shape[2], dist_q_target_t.q_dist.shape[3]),
            )

            # Batch x Ensemble
            repeated_actions = jnp.repeat(transitions.a_tm1, ens_size)
            repeated_discounts = jnp.repeat(transitions.discount_t, ens_size)

            if weights is not None:
                repeated_weights = jnp.repeat(weights, ens_size)

            if transitions.r_t.shape == repeated_actions.shape:
                repeated_rewards = transitions.r_t
            else:
                repeated_rewards = jnp.repeat(transitions.r_t, ens_size)

            losses = _batch_quantile_q_learning(
                flattened_dist_q_tm1,
                quantiles,
                repeated_actions,
                repeated_rewards,
                repeated_discounts,
                flattened_dist_q_target_t,  # No double Q-learning here.
                flattened_dist_q_target_t,
                huber_param,
            )

            chex.assert_shape((losses), (self._batch_size * ens_size,))
            if weights is not None:
                chex.assert_shape((repeated_weights), (self._batch_size * ens_size,))

            mask = jax.lax.stop_gradient(jnp.reshape(transitions.mask_t, (-1,)))
            loss = mask * losses

            if weights is not None:
                loss = repeated_weights * loss

            loss = jnp.sum(loss) / jnp.sum(mask)

            # compute logging quantities
            # mean over actions
            mean_q = jnp.mean(dist_q_tm1.q_values, axis=1)
            mean_q_var = jnp.mean(dist_q_tm1.q_values_var, axis=1)
            mean_epistemic = jnp.mean(dist_q_tm1.epistemic_uncertainty, axis=1)
            mean_aleatoric = jnp.mean(dist_q_tm1.aleatoric_uncertainty, axis=1)

            # quantities of action chosen
            q_select = _select_actions(dist_q_tm1.q_values, transitions.a_tm1)
            q_var_select = _select_actions(dist_q_tm1.q_values_var, transitions.a_tm1)
            epistemic_select = _select_actions(
                dist_q_tm1.epistemic_uncertainty, transitions.a_tm1
            )
            aleatoric_select = _select_actions(
                dist_q_tm1.aleatoric_uncertainty, transitions.a_tm1
            )

            td_errors = jnp.mean(losses.reshape((-1, ens_size)), axis=1)

            return loss, {
                "loss": loss,
                "td_errors": td_errors,
                "mean_q": mean_q,
                "mean_q_var": mean_q_var,
                "mean_epistemic": mean_epistemic,
                "mean_aleatoric": mean_aleatoric,
                "q_select": q_select,
                "q_var_select": q_var_select,
                "epistemic_select": epistemic_select,
                "aleatoric_select": aleatoric_select,
            }

        def update(
            rng_key,
            opt_state,
            online_params,
            target_params,
            transitions,
            weights,
            stop_grad,
        ):
            """Computes learning update from batch of replay transitions."""
            rng_key, update_key = jax.random.split(rng_key)
            d_loss_d_params, aux = jax.grad(loss_fn, has_aux=True)(
                online_params,
                target_params,
                transitions,
                weights,
                update_key,
                stop_grad,
            )
            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            new_online_params = optax.apply_updates(online_params, updates)
            return rng_key, new_opt_state, new_online_params, aux

        self._update = jax.jit(update)

        def select_action(rng_key, network_params, s_t, exploration_epsilon):
            """Samples action from eps-greedy policy wrt Q-values at given state."""
            rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)

            network_forward = network.apply(
                network_params, apply_key, (s_t[None, ...], False)
            )

            q_t = network_forward.q_values[0]  # average of multi-head output

            a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(
                seed=policy_key
            )

            # if a_t == -1:
            #     import pdb

            #     pdb.set_trace()
            # a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(
            #     seed=policy_key
            # )
            # print("action", a_t)
            v_t = jnp.max(q_t, axis=-1)
            return rng_key, a_t, v_t

        # self._select_action = select_action
        self._select_action = jax.jit(select_action)

    def _get_random_mask(self, rng_key):
        return jax.random.choice(
            key=rng_key, a=2, shape=(self._ens_size,), p=self._mask_probabilities
        )

    def step(self, timestep: dm_env.TimeStep) -> parts.Action:
        """Selects action given timestep and potentially learns."""
        self._frame_t += 1

        timestep = self._preprocessor(timestep)

        if timestep is None:  # Repeat action.
            action = self._action
        else:
            action = self._action = self._act(timestep)

            for transition in self._transition_accumulator.step(timestep, action):
                mask = self._get_random_mask(self._rng_key)
                transition = replay_lib.MaskedTransition(
                    s_tm1=transition.s_tm1,
                    a_tm1=transition.a_tm1,
                    r_t=transition.r_t,
                    discount_t=transition.discount_t,
                    s_t=transition.s_t,
                    mask_t=mask,
                )
                if self._prioritise:
                    self._replay.add(transition, priority=self._max_seen_priority)
                else:
                    self._replay.add(transition)

        if self._replay.size < self._min_replay_capacity:
            return action, {}

        if self._frame_t % self._learn_period == 0:
            aux = self._learn()
        else:
            aux = {}

        if self._frame_t % self._target_network_update_period == 0:
            self._target_params = self._online_params

        return action, aux

    def reset(self) -> None:
        """Resets the agent's episodic state such as frame stack and action repeat.

        This method should be called at the beginning of every episode.
        """
        self._transition_accumulator.reset()
        processors.reset(self._preprocessor)
        self._action = None

    def _act(self, timestep) -> parts.Action:
        """Selects action given timestep, according to epsilon-greedy policy."""
        s_t = timestep.observation
        self._rng_key, a_t, v_t = self._select_action(
            self._rng_key, self._online_params, s_t, self.exploration_epsilon
        )
        a_t, v_t = jax.device_get((a_t, v_t))
        self._statistics["state_value"] = v_t
        return parts.Action(a_t)

    def _learn(self) -> None:
        """Samples a batch of transitions from replay and learns from it."""
        logging.log_first_n(logging.INFO, "Begin learning", 1)
        random_head = np.random.choice(range(self._ens_size))
        if self._prioritise is not None:
            for i in range(self._ens_size):
                transitions, indices, weights = self._replay.sample(self._batch_size)
                self._rng_key, self._opt_state, self._online_params, aux = self._update(
                    self._rng_key,
                    self._opt_state,
                    self._online_params,
                    self._target_params,
                    transitions,
                    weights,
                    stop_grad=i != random_head,
                )
        else:
            for i in range(self._ens_size):
                transitions = self._replay.sample(self._batch_size)
                self._rng_key, self._opt_state, self._online_params, aux = self._update(
                    self._rng_key,
                    self._opt_state,
                    self._online_params,
                    self._target_params,
                    transitions,
                    None,
                    stop_grad=i != random_head,
                )

        if self._prioritise is not None:
            if self._prioritise == "averaged_uncertainty":
                chex.assert_equal_shape((weights, aux["mean_epistemic"]))
                priorities = jnp.abs(aux["mean_epistemic"])
                priorities = jax.device_get(priorities)
            if self._prioritise == "uncertainty":
                chex.assert_equal_shape((weights, aux["epistemic_select"]))
                priorities = jnp.abs(aux["epistemic_select"])
                priorities = jax.device_get(priorities)
            elif self._prioritise == "uncertainty_ratio":
                chex.assert_equal_shape((weights, aux["epistemic_select"]))
                chex.assert_equal_shape((weights, aux["aleatoric_select"]))
                priorities = jnp.abs(
                    aux["mean_epistemic"]
                    / (aux["mean_epistemic"] + aux["mean_aleatoric"])
                )
                priorities = jax.device_get(priorities)
            elif self._prioritise == "uncertainty_ratio_select":
                chex.assert_equal_shape((weights, aux["epistemic_select"]))
                chex.assert_equal_shape((weights, aux["aleatoric_select"]))
                priorities = jnp.abs(
                    aux["epistemic_select"]
                    / (aux["epistemic_select"] + aux["aleatoric_select"])
                )
                priorities = jax.device_get(priorities)
            elif self._prioritise == "td":
                chex.assert_equal_shape((weights, aux["td_errors"]))
                priorities = jnp.abs(aux["td_errors"])
            priorities = jax.device_get(priorities)
            max_priority = priorities.max()
            self._max_seen_priority = np.max([self._max_seen_priority, max_priority])
            self._replay.update_priorities(indices, priorities)

        aux = {k: jnp.mean(v) for k, v in aux.items()}

        return aux

    @property
    def online_params(self) -> parts.NetworkParams:
        """Returns current parameters of Q-network."""
        return self._online_params

    @property
    def statistics(self) -> Mapping[Text, float]:
        """Returns current agent statistics as a dictionary."""
        # Check for DeviceArrays in values as this can be very slow.
        assert all(
            not isinstance(x, jnp.DeviceArray) for x in self._statistics.values()
        )
        return self._statistics

    @property
    def exploration_epsilon(self) -> float:
        """Returns epsilon value currently used by (eps-greedy) behavior policy."""
        return self._exploration_epsilon(self._frame_t)

    def get_state(self) -> Mapping[Text, Any]:
        """Retrieves agent state as a dictionary (e.g. for serialization)."""
        state = {
            "rng_key": self._rng_key,
            "frame_t": self._frame_t,
            "opt_state": self._opt_state,
            "online_params": self._online_params,
            "target_params": self._target_params,
            "replay": self._replay.get_state(),
        }
        return state

    def set_state(self, state: Mapping[Text, Any]) -> None:
        """Sets agent state from a (potentially de-serialized) dictionary."""
        self._rng_key = state["rng_key"]
        self._frame_t = state["frame_t"]
        self._opt_state = jax.device_put(state["opt_state"])
        self._online_params = jax.device_put(state["online_params"])
        self._target_params = jax.device_put(state["target_params"])
        self._replay.set_state(state["replay"])
