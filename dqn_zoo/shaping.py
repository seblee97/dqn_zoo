import jax.numpy as jnp
import jax


class NoPenalty:
  """No penalty placeholder."""

  def __call__(self, target_q_values, transitions, rng_key):
    return transitions.r_t


class HardCodedPenalty:
  """Hard coded constant penalty,
  i.e. F(s, a, s') = k where k is constant.
  """

  def __init__(self, penalty: float):
      self._penalty = penalty

  def __call__(self, target_q_values, transitions, rng_key):

    penalty_terms = self._penalty * jnp.ones_like(transitions.r_t)
    
    return transitions.r_t + penalty_terms
      
class UncertaintyPenalty:
  """Adaptive penalty based on uncertainty in state-action values over ensemble."""

  def __init__(self, multiplicative_factor: float):
    self._multiplicative_factor = multiplicative_factor

  def __call__(self, target_q_values, transitions, rng_key):
    state_action_values = target_q_values[jnp.arange(len(target_q_values)), :, transitions.a_tm1]
    penalty_terms = self._multiplicative_factor * jnp.std(state_action_values, axis=1)
    return transitions.r_t + penalty_terms

class PolicyEntropyPenalty:
  """Adaptive penalty based on policy entropy of ensemble."""

  def __init__(self, multiplicative_factor: float, num_actions: int):
    self._multiplicative_factor = multiplicative_factor
    LOG_EPSILON = 0.0001

    def compute_entropy(max_indices):
      max_index_probabilities = jnp.bincount(max_indices, minlength=num_actions, length=num_actions) / len(max_indices)
      entropy = -jnp.sum((max_index_probabilities + LOG_EPSILON) * jnp.log(max_index_probabilities + LOG_EPSILON))
      return entropy

    self._compute_entropy = jax.vmap(compute_entropy, in_axes=(0))

  def __call__(self, target_q_values, transitions, rng_key):
    max_indices = jnp.argmax(target_q_values, axis=-1)
    penalty_terms = self._multiplicative_factor * self._compute_entropy(max_indices)
    return transitions.r_t + penalty_terms
    
class MunchausenPenalty:
  """Adaptive penalty that adds scaled log policy to the reward.
  
  Based on M-IQN thinking in Munchausen RL: https://arxiv.org/pdf/2007.14430.pdf
  """

  def __init__(self, multiplicative_factor: float, num_actions: int):
    self._multiplicative_factor = multiplicative_factor
    LOG_EPSILON = 0.0001

    def compute_log_policy(max_indices):
      max_index_probabilities = jnp.bincount(max_indices, minlength=num_actions, length=num_actions) / len(max_indices)
      log_policy = jnp.log(max_index_probabilities + LOG_EPSILON)
      return log_policy

    self._compute_log_policy = jax.vmap(compute_log_policy, in_axes=(0))

  def __call__(self, target_q_values, transitions, rng_key):
    max_indices = jnp.argmax(target_q_values, axis=-1)
    log_policy = self._compute_log_policy(max_indices)
    action_log_policy = log_policy[jnp.arange(len(log_policy)), transitions.a_tm1]

    penalty_terms = self._multiplicative_factor * action_log_policy

    return transitions.r_t + penalty_terms
