"""Implementation of bootstrapped DQN (Osband 2016)"""

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

# Batch variant of q_learning.
_batch_q_learning = jax.vmap(rlax.q_learning)
_batch_double_q_learning = jax.vmap(rlax.double_q_learning)


class BootstrappedDqn(parts.Agent):
    """Deep Q-Network agent with multiple heads."""

    def __init__(
        self,
        preprocessor: processors.Processor,
        sample_network_input: jnp.ndarray,
        network: parts.Network,
        variance_network: bool,
        optimizer: optax.GradientTransformation,
        transition_accumulator: Any,
        replay: replay_lib.PrioritizedTransitionReplay,
        shaping,
        mask_probability: float,
        num_heads: int,
        batch_size: int,
        exploration_epsilon: Callable[[int], float],
        min_replay_capacity_fraction: float,
        learn_period: int,
        target_network_update_period: int,
        grad_error_bound: float,
        rng_key: parts.PRNGKey,
    ):
        self._preprocessor = preprocessor
        self._replay = replay
        self._transition_accumulator = transition_accumulator
        self._mask_probabilities = jnp.array([mask_probability, 1 - mask_probability])
        self._num_heads = num_heads
        self._batch_size = batch_size
        self._exploration_epsilon = exploration_epsilon
        self._min_replay_capacity = min_replay_capacity_fraction * replay.capacity
        self._learn_period = learn_period
        self._target_network_update_period = target_network_update_period
        self._variance_network = variance_network

        # Initialize network parameters and optimizer.
        self._rng_key, network_rng_key, var_network_rng_key = jax.random.split(
            rng_key, 3
        )

        self._online_params = network.init(
            network_rng_key, sample_network_input[None, ...]
        )
        self._target_params = self._online_params
        self._opt_state = optimizer.init(self._online_params)

        if variance_network:
            self._var_online_params = network.init(
                var_network_rng_key, sample_network_input[None, ...]
            )
            self._var_target_params = self._var_online_params
            self._var_opt_state = optimizer.init(self._var_online_params)
        else:
            self._var_online_params = None
            self._var_target_params = None
            self._var_opt_state = None

        # Other agent state: last action, frame count, etc.
        self._action = None
        self._frame_t = -1  # Current frame index.
        self._statistics = {"state_value": np.nan}
        self._max_seen_priority = 1.

        LOG_EPSILON = 0.0001

        # Define jitted loss, update, and policy functions here instead of as
        # class methods, to emphasize that these are meant to be pure functions
        # and should not access the agent object's state via `self`.

        def _forward(params, state):
            _, split_key = jax.random.split(rng_key)
            q_values = network.apply(params, split_key, state).multi_head_output
            return q_values

        self._forward = jax.jit(_forward)

        def shaping_output(target_params, transitions, rng_key):
            _, *apply_keys = jax.random.split(rng_key, 3)
            q_target_t = network.apply(
                target_params, apply_keys[0], transitions.s_t
            ).multi_head_output
            flattened_q_target = jnp.reshape(q_target_t, (-1, q_target_t.shape[-1]))
            # compute shaping function F(s, a, s')
            shaped_rewards = shaping_function(q_target_t, transitions, apply_keys[1])
            penalties = shaped_rewards - transitions.r_t

            return q_target_t, flattened_q_target, shaped_rewards, penalties

        def loss_fn(online_params, target_params, transitions, rng_key):
            """Calculates loss given network parameters and transitions."""
            _, *apply_keys = jax.random.split(rng_key, 4)
            # Batch : Heads : Actions (source state)
            q_tm1 = network.apply(
                online_params, apply_keys[0], transitions.s_tm1
            ).multi_head_output
            # Batch : Heads : Actions (destination state)
            q_t = network.apply(
                online_params, apply_keys[1], transitions.s_t
            ).multi_head_output
            # Batch : Heads : Actions (destination state, target network)
            q_target_t = network.apply(
                target_params, apply_keys[2], transitions.s_t
            ).multi_head_output

            # Batch x Heads : Actions
            flattened_q = jnp.reshape(q_tm1, (-1, q_tm1.shape[-1]))
            flattened_q_t = jnp.reshape(q_t, (-1, q_t.shape[-1]))
            flattened_q_target = jnp.reshape(q_target_t, (-1, q_target_t.shape[-1]))

            # Batch x Heads
            repeated_actions = jnp.repeat(transitions.a_tm1, num_heads)
            repeated_discounts = jnp.repeat(transitions.discount_t, num_heads)

            if transitions.r_t.shape == repeated_actions.shape:
                repeated_rewards = transitions.r_t
            else:
                repeated_rewards = jnp.repeat(transitions.r_t, num_heads)

            # Batch x Heads
            raw_td_errors = _batch_double_q_learning(
                flattened_q,
                repeated_actions,
                repeated_rewards,
                repeated_discounts,
                flattened_q_target,
                flattened_q_t,
            )

            clipped_td_errors = rlax.clip_gradient(
                raw_td_errors,
                -grad_error_bound / num_heads,
                grad_error_bound / num_heads,
            )
            losses = rlax.l2_loss(clipped_td_errors)
            assert losses.shape == (self._batch_size * num_heads,)

            mask = jax.lax.stop_gradient(jnp.reshape(transitions.mask_t, (-1,)))
            loss = jnp.sum(mask * losses) / jnp.sum(mask)
            return loss, {"loss": loss, "deltas": raw_td_errors}

        def update(
            rng_key,
            opt_state,
            online_params,
            target_params,
            transitions,
            var_opt_state,
            var_online_params,
            var_target_params,
        ):
            transitions = transitions[0]  # PATH FIXING
            """Computes learning update from batch of replay transitions."""
            rng_key, update_key = jax.random.split(rng_key)
            d_loss_d_params, aux = jax.grad(loss_fn, has_aux=True)(
                online_params,
                target_params,
                transitions,
                update_key,
            )

            updates, new_opt_state = optimizer.update(d_loss_d_params, opt_state)
            new_online_params = optax.apply_updates(online_params, updates)

            if variance_network:
                # create identical transition object with meta-reward
                var_transitions = replay_lib.MaskedTransition(
                    s_tm1=transitions.s_tm1,
                    a_tm1=transitions.a_tm1,
                    r_t=aux["deltas"] ** 2,
                    discount_t=transitions.discount_t**2,
                    s_t=transitions.s_t,
                    mask_t=transitions.mask_t,
                )

                rng_key, var_update_key = jax.random.split(rng_key)
                var_d_loss_d_params, var_aux = jax.grad(loss_fn, has_aux=True)(
                    var_online_params,
                    var_target_params,
                    var_transitions,
                    var_update_key,
                )

                var_updates, var_new_opt_state = optimizer.update(
                    var_d_loss_d_params, var_opt_state
                )
                var_new_online_params = optax.apply_updates(
                    var_online_params, var_updates
                )
            else:
                var_new_opt_state = None
                var_new_online_params = None

            return (
                rng_key,
                new_opt_state,
                new_online_params,
                var_new_opt_state,
                var_new_online_params,
            )

        # self._update = update
        self._update = jax.jit(update)

        def select_action(rng_key, network_params, s_t, exploration_epsilon):
            """Samples action from eps-greedy policy wrt Q-values at given state."""
            rng_key, apply_key, policy_key = jax.random.split(rng_key, 3)

            network_forward = network.apply(network_params, apply_key, s_t[None, ...])
            q_t = network_forward.q_values[0]  # average of multi-head output

            a_t = distrax.EpsilonGreedy(q_t, exploration_epsilon).sample(
                seed=policy_key
            )
            v_t = jnp.max(q_t, axis=-1)
            return rng_key, a_t, v_t

        self._select_action = jax.jit(select_action)
        # self._select_action = select_action

    def _get_random_mask(self, rng_key):
        return jax.random.choice(
            key=rng_key, a=2, shape=(self._num_heads,), p=self._mask_probabilities
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
                self._replay.add(transition, priority=self._max_seen_priority)

        if self._replay.size < self._min_replay_capacity:
            return action, None, None, None

        if self._frame_t % self._learn_period == 0:
            # loss, shaped_rewards, penalties = self._learn()
            self._learn()
            loss = None
            shaped_rewards = None
            penalties = None
        else:
            loss = None
            shaped_rewards = None
            penalties = None

        if self._frame_t % self._target_network_update_period == 0:
            self._target_params = self._online_params

        return action, loss, shaped_rewards, penalties

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
        transitions = self._replay.sample(self._batch_size)
        (
            self._rng_key,
            self._opt_state,
            self._online_params,
            self._var_opt_state,
            self._var_online_params
            # loss_values,
            # shaped_rewards,
            # penalties,
        ) = self._update(
            self._rng_key,
            self._opt_state,
            self._online_params,
            self._target_params,
            transitions,
            self._var_opt_state,
            self._var_online_params,
            self._var_target_params,
        )
        # return loss_values.item(), shaped_rewards.tolist(), penalties.tolist()

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

    def forward_all_heads(self, state, variance: bool = False):
        if variance:
            params = self._var_online_params
        else:
            params = self._online_params
        return self._forward(state=state, params=params)
