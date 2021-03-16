import jax.numpy as jnp


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
    pass

class PolicyEntropyPenalty:
  """Adaptive penalty based on policy entropy of ensemble."""

  def __init__(self, multiplicative_factor: float):
    self._multiplicative_factor = multiplicative_factor

  def __call__(self, target_q_values, transitions, rng_key):
    pass
