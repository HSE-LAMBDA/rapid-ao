import numpy as np

from .meta import Task
from .utils import rot_matrix

__all__ = [
  'XOR'
]

def get_petals(size):
    data = np.random.normal(size=(size, 2))
    data = np.dot(data, np.array([[0.25, 0.5], [0.5, 0.25]]))
    data += np.array([2, 2])[None, :] * (1 - 2 * np.random.binomial(n=1, p=0.5, size=(size,)))[:, None]
    return data

class XOR(Task):
  def __init__(self):
    super(XOR, self).__init__(ndim=2)

  def search_space(self):
    return [
      [-np.pi / 16, np.pi / 2]
    ]

  def solution(self):
    return np.array([0.0], dtype='float32')

  def example_parameters(self):
    return np.array([np.pi / 2])

  def parameters_names(self):
    return ['rotation angle']

  def ground_truth_generator(self):
    return get_petals

  def generator(self, params):
    angle = params[0]
    return lambda size: np.dot(get_petals(size), rot_matrix(angle))

  def transform(self, data0, params):
    angle = params[0]
    return np.dot(data0, rot_matrix(angle))

  def model_parameters(self):
    return dict(
      n_estimators=100,
      max_depth=3,
      x0_lin=np.log(2) / 4,
      x0_log=np.log(2) / 4,

      n_units=128,
      grad_penalty=5e-2,
    )