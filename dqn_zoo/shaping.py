import jax.numpy as jnp


class NoPenalty:
  """No penalty placeholder."""

  def __call__(self, target_params, transitions, rng_key):
    return transitions.r_t


class HardCodedPenalty:
  """Hard coded constant penalty,
  i.e. F(s, a, s') = k where k is constant.
  """

  def __init__(self, penalty: float):
      self._penalty = penalty

  def __call__(self, target_params, transitions, rng_key):

    penalty_terms = self._penalty * jnp.ones_like(transitions.r_t)
    
    return transitions.r_t + penalty_terms
      