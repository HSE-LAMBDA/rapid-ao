from collections import namedtuple

import numpy as np

OptimizationResults = namedtuple('OptimizationResults', [
  'points', 'values', 'costs', 'seed'
])

def transform(x, space):
  space = np.asarray(space)
  return (x - space[None, :, 0]) / (space[:, 1] - space[:, 0])[None, :]

def reverse_transform(x, space):
  space = np.asarray(space)
  return x * (space[:, 1] - space[:, 0])[None, :] + space[None, :, 0]

class Optimizer(object):
  def __init__(self, space, seed = None):
    self.space = np.asarray(space)
    self.models = None

    self.rng = np.random.RandomState(seed=seed)

  def name(self):
    return self.__class__.__name__

  def ask(self):
    raise NotImplementedError()

  def tell(self, x, y):
    raise NotImplementedError()

  def local_method(self):
    """
    Determines if classifier should be reset to fit parameters.
    Local in this context means that Optimizer returns close suggestions
    on consecutive iterations of the ask-tell cycle.
    :return: bool
    """
    raise NotImplementedError()