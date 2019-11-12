import numpy as np

from .meta import Task
from .utils import rot_matrix

__all__ = [
  'SwissRoll'
]

def get_roll(size, noise=0.05):
  """
  Stolen from https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/datasets/samples_generator.py#L1320
  """
  t = 1.5 * np.pi * (1 + 2 * np.random.rand(size, 1))
  x = t * np.cos(t) / 10 + np.random.normal(scale=noise, size=(size, 1))
  y = t * np.sin(t) / 10 + np.random.normal(scale=noise, size=(size, 1))

  return np.concatenate([x, y], axis=1)

class SwissRoll(Task):
  def __init__(self, noise=0.05, seed=None):
    super(SwissRoll, self).__init__(ndim=2, seed=seed)
    self.noise = noise

  def search_space(self):
    return [
      [-np.pi / 16, np.pi]
    ]

  def solution(self):
    return np.array([0.0], dtype='float32')

  def example_parameters(self):
    return np.array([np.pi / 2])

  def parameters_names(self):
    return ['rotation angle']

  def ground_truth_generator(self):
    return lambda size: get_roll(size, self.noise)

  def generator(self, params):
    if callable(params):
      def gen(size):
        ps = params(size)
        return (
          np.concatenate([
            np.dot(point.reshape(1, 2), rot_matrix(angle[0]))
            for point, angle in zip(get_roll(size, self.noise), params(size))
          ], axis=0),
          ps,
        )
      return gen
    else:
      angle = params[0]
      return lambda size: (
        np.dot(get_roll(size, self.noise), rot_matrix(angle)),
        np.repeat(params[None, :], repeats=size, axis=0)
      )

  def transform(self, data0, params):
    angle = params[0]
    return np.dot(data0, rot_matrix(angle))

  def model_parameters(self):
    return dict(
      n_estimators=100,
      max_depth=3,

      n_units=256,
      grad_penalty=5e-2,
    )