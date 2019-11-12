import numpy as np

from .meta import OptimizationResults

from ..target.meta import cached_generator
from ..target import fixed_size

__all__ = [
  'grid_search',
]

def make_grid(space, steps):
  xs = tuple([
    np.linspace(lower_x, upper_x, num=steps)
    for lower_x, upper_x in space
  ])

  grids = np.meshgrid(*xs)
  return np.vstack([
    grid.reshape(-1)
    for grid in grids
  ]).T


class grid_search(object):
  def __init__(self, steps, metric):
    self.steps = steps
    if isinstance(metric, int):
      self.metric = fixed_size(metric)
    else:
      self.metric = metric

  def name(self):
    return 'GridSearch'

  def __call__(self, task, model, budget, seed=None):
    grid = make_grid(task.search_space(), self.steps)

    master_rng = np.random.RandomState(seed=seed)
    get_seed = lambda: master_rng.randint(0, 2 ** 31, dtype='int32')

    task.set_seed(get_seed())
    ground_truth_generator = task.ground_truth_generator()

    ### to avoid spawning two generator instances
    gen_pos_train = cached_generator(ground_truth_generator)
    gen_pos_val = cached_generator(ground_truth_generator)

    known_points = []
    known_values = []
    costs = []

    for point in grid:
      gen_neg = task.generator(point)

      ### to avoid spawning two generator instances
      gen_neg_train = cached_generator(gen_neg)
      gen_neg_val = cached_generator(gen_neg)

      cost, value = self.metric(model, gen_pos_train, gen_neg_train, gen_pos_val, gen_neg_val)

      known_points.append(point)
      known_values.append(value)
      costs.append(cost)

    return OptimizationResults(
      points=known_points,
      values=known_values,
      costs=costs,
      seed=seed
    )

