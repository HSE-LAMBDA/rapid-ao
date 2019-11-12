import inspect
import numpy as np

from skopt import Optimizer as BOOptimizer

from advopt.target.meta import cached_generator
from advopt.opt.meta import OptimizationResults

from advopt.target import adjusted

__all__ = [
  'bayesian_optimization',
]

BO_signature = inspect.signature(BOOptimizer)

class bayesian_optimization(object):
  def __init__(self, *args, metric=adjusted(), **kwargs):
    self.args = args
    self.kwargs = kwargs
    self.metric = metric

    ### assert that arguments are correct
    bayesian_optimization.__signature__.bind(
      *self.args, metric=metric, **self.kwargs
    )

  def name(self):
    return 'BayesianOptimization'

  def __call__(self, task, model, budget, seed=None, callback=None):
    if callback is None:
      callback = lambda optimizer: None

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
    total_cost = 0

    optimizer = BOOptimizer(
      *self.args,
      dimensions=task.search_space(),
      random_state=get_seed(),
      **self.kwargs
    )

    while True:
      if budget is not None and total_cost >= budget:
        break

      try:
        point = optimizer.ask()
      except StopIteration:
        break

      gen_neg = task.generator(np.array(point, dtype='float64'))

      ### to avoid spawning two generator instances
      gen_neg_train = cached_generator(gen_neg)
      gen_neg_val = cached_generator(gen_neg)

      if budget is not None:
        left = budget - total_cost
      else:
        left = None

      cost, value = self.metric(model, gen_pos_train, gen_neg_train, gen_pos_val, gen_neg_val, budget=left)
      if cost is None or value is None:
        return OptimizationResults(
          points=known_points,
          values=known_values,
          costs=costs,
          seed=seed
        )

      optimizer.tell(point, value)

      callback(optimizer)

      known_points.append(point)
      known_values.append(value)
      costs.append(cost)

      total_cost += cost

    return OptimizationResults(
      points=known_points,
      values=known_values,
      costs=costs,
      seed=seed
    )


bayesian_optimization.__signature__ = inspect.Signature(
  parameters=[
    param for name, param in BO_signature.parameters.items()
    if name != 'random_state' and name != 'dimensions'
  ] + [
    inspect.Parameter('metric', inspect.Parameter.KEYWORD_ONLY, default=adjusted())
  ],
  return_annotation=BO_signature.return_annotation
)