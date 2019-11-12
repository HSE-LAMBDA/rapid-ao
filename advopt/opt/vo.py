import inspect

import numpy as np
import torch
from torch.nn import functional as F

from advopt.target.meta import cached_generator
from advopt.opt.meta import OptimizationResults

from advopt.target import adjusted

from .meta import Optimizer

__all__ = [
  'variational_optimization',
]

class VariationalOptimizer(Optimizer):
  def __init__(self, space, mean0=None, sigma=None, sigma0=None, min_sigma=None, device='cpu', seed=None, learning_rate=1e-2, **optimizer_kwargs):
    """
    :param space: constraints on optimized parameters
    :param sigma: if None, then variance (root of variance) of the search distribution is a free parameters,
      otherwise it is set to the value of `sigma`. Free `sigma` is internally represented as:
      `lop(1 + exp(sigma')) + min_sigma`
    :param sigma0: initial value for variance of the search distribution,
      if the latter is a free variable.
    :param min_sigma: restrictions on sigma variable, if latter is a free variable.
      If `min_sigma` is None, then the variance is restricted to [0, +inf).
    :param device: see devices in torch
    :param seed:
    :param learning_rate:
    :param optimizer_kwargs:
    """
    super(VariationalOptimizer, self).__init__(space, seed=seed)

    self.sigma = None

    if mean0 is None:
      ### start at the center of space
      mean0 = (self.space[:, 1] + self.space[:, 0]) / 2
    else:
      mean0 = np.asarray(mean0, dtype='float32')

    self.mean = torch.tensor(mean0, dtype=torch.float32, device=device, requires_grad=True)

    if sigma is None:
      if sigma0 is None:
        ### start with quarter of the space span.
        sigma0 = (self.space[:, 1] - self.space[:, 0]) / 4

      if min_sigma is None:
        sigma_raw_0 = np.log(np.exp(sigma0) - 1)
      else:
        sigma_raw_0 = np.log(np.exp(sigma0) - 1) - min_sigma

      ### sigma := log(1 + exp(sigma_raw)), this way sigma_raw has no restrictions
      self.sigma_raw = torch.tensor(sigma_raw_0, dtype=torch.float32, device=device, requires_grad=True)

      if min_sigma is None:
        self.sigma = lambda : F.softplus(self.sigma_raw)
      else:
        self.sigma = lambda : F.softplus(self.sigma_raw) + min_sigma

      self.params = [self.mean, self.sigma_raw]

    else:
      assert sigma > 0
      self.sigma_raw = torch.tensor(np.float32(sigma), dtype=torch.float32, device=device, requires_grad=False)
      self.sigma = lambda : self.sigma_raw
      self.params = [self.mean]

    self.ndim = len(space)
    self.device = device

    self.optimizer = torch.optim.Adam(params=self.params, lr=learning_rate, **optimizer_kwargs)

  def name(self):
    return 'VariationalOptimization'

  def ask(self):
    mean = self.mean.detach().cpu().numpy()
    std = self.sigma().detach().cpu().numpy()

    return mean, std

  def tell(self, params, target):
    X = torch.tensor(params, dtype=torch.float32, device=self.device, requires_grad=False)
    y = torch.tensor(target, dtype=torch.float32, device=self.device, requires_grad=False)

    self.optimizer.zero_grad()

    sigma = self.sigma()
    log_p = self.ndim * torch.log(sigma) + 0.5 * torch.sum((X - self.mean[None, :]) ** 2, dim=1) / (sigma ** 2)
    loss = -torch.mean(y * log_p)

    loss.backward()

    self.optimizer.step()

class variational_optimization(object):
  def   __init__(
    self, metric=adjusted(),
    mean0=None, sigma=None, sigma0=None, sigma_min=None,
    device='cpu', learning_rate=1e-2, **optimizer_kwargs
  ):
    self.metric = metric

    self.mean0 = mean0

    self.sigma = sigma
    self.sigma0 = sigma0
    self.sigma_min = sigma_min

    self.learning_rate = learning_rate
    self.kwargs = optimizer_kwargs

    self.device = device

  def name(self):
    return 'VariationalOptimization'

  def __call__(self, task, budget, model, seed):
    ndim = len(task.search_space())

    master_rng = np.random.RandomState(seed=seed)
    get_seed = lambda: master_rng.randint(0, 2 ** 31, dtype='int32')

    optimizer = VariationalOptimizer(
      space=task.search_space(),
      mean0=self.mean0,
      sigma=self.sigma, sigma0=self.sigma0,
      device=self.device,
      learning_rate=self.learning_rate,
      seed=get_seed(),
      **self.kwargs
    )
    task.set_seed(get_seed())
    ground_truth_generator = task.ground_truth_generator()

    ### to avoid spawning two generator instances
    gen_pos_train = cached_generator(ground_truth_generator)
    gen_pos_val = cached_generator(ground_truth_generator)

    known_points = []
    known_values = []
    costs = []

    total_cost = 0

    while True:
      if budget is not None and total_cost >= budget:
        break

      try:
        mean, std = optimizer.ask()
      except StopIteration:
        break

      print('mean:', mean, 'sigma:', std)
      gen_neg = task.generator(
        lambda size: np.random.normal(size=(size, ndim)) * std + mean[None, :]
      )

      ### to avoid spawning two generator instances
      gen_neg_train = cached_generator(gen_neg)
      gen_neg_val = cached_generator(gen_neg)

      if budget is None:
        left = None
      else:
        left = budget - total_cost

      cost, value = self.metric(
        model,
        gen_pos_train, gen_neg_train,
        gen_pos_val, gen_neg_val,
        budget=left
      )

      if cost is None:
        return OptimizationResults(
          points=known_points,
          values=known_values,
          costs=costs,
          seed=seed
        )

      ### generator is cached, therefore, it does not generate new samples.
      X_neg_val, params_neg_val = gen_neg_val(cost)

      proba_neg_val = model.predict_proba(X_neg_val)[:, 1]
      jsd_neg_val = np.log(1 - proba_neg_val + 1e-6)

      optimizer.tell(
        params=params_neg_val,
        target=jsd_neg_val
      )

      known_points.append(np.copy(mean))
      known_values.append(value)
      costs.append(cost)

      total_cost += cost

    return OptimizationResults(
      points=known_points,
      values=known_values,
      costs=costs,
      seed=seed
    )