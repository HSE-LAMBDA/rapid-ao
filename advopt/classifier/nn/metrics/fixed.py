import torch

from ....target import cached_generators, metric
from ..meta import Network


class fixed_size(metric):
  def __init__(
    self, size, n_iterations='adaptive'
  ):
    """
    :param size: size of the data for training (and validation) datasets
    :param n_iterations: either:
      - an integer - train for `n_iterations` iterations;
      - 'convergence' - train until convergence;
      - 'adaptive' - the same as 'convergence', but with early stop due to
        high train/validation loss difference. Works only if `Network` has `stop_diff` argument set.
    """
    super(fixed_size, self).__init__()
    self.size = size
    self.n_iterations = n_iterations

  def __call__(self, clf : Network, gen_pos, gen_neg, gen_pos_val=None, gen_neg_val=None, budget=None):
    gen_pos, gen_pos_val = cached_generators(gen_pos, gen_pos_val)
    gen_neg, gen_neg_val = cached_generators(gen_neg, gen_neg_val)

    X_pos = torch.tensor(
      gen_pos.samples(self.size), dtype=torch.float32,
      device=clf.device, requires_grad=False
    )
    X_neg = torch.tensor(
      gen_neg.samples(self.size), dtype=torch.float32,
      device=clf.device, requires_grad=False
    )

    if isinstance(self.n_iterations, int):
      ce = clf.fixed_fit(X_pos, X_neg, self.n_iterations)
      return self.size, ce

    elif self.n_iterations == 'convergence':
      ce = clf.adaptive_fit(X_pos, X_neg)
      return self.size, ce

    elif self.n_iterations == 'adaptive':
      X_pos_val = torch.tensor(
        gen_pos_val.samples(self.size), dtype=torch.float32,
        device=clf.device, requires_grad=False
      )
      X_neg_val = torch.tensor(
        gen_neg_val.samples(self.size), dtype=torch.float32,
        device=clf.device, requires_grad=False
      )

      ce, ce_val = clf.adaptive_fit(X_pos, X_neg, X_pos_val, X_neg_val)
      if ce is not None:
        return self.size, (ce + ce_val) / 2
      else:
        return self.size, None


