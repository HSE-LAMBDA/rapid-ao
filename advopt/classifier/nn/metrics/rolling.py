import numpy as np
import torch

from advopt.target.adjusted import cached_generators, metric
from advopt.classifier.nn import Network


__all__ = [
  'rolling'
]

class rolling(metric):
  def __init__(
    self, size_step=128, target_diff=1e-2, progress=None,
  ):
    super(rolling, self).__init__()

    self.size_step = size_step
    self.progress = progress
    self.target_diff = target_diff

  def __call__(self, clf : Network, gen_pos, gen_neg, gen_pos_val=None, gen_neg_val=None, budget=None):
    clf.soft_reset()

    gen_pos, gen_pos_val = cached_generators(gen_pos, gen_pos_val)
    gen_neg, gen_neg_val = cached_generators(gen_neg, gen_neg_val)

    data_real = gen_pos(budget)[0]
    data_real_val = gen_pos(budget)[0]

    X_pos = torch.tensor(data_real, dtype=torch.float32, device=clf.device, requires_grad=False)
    X_pos_val = torch.tensor(data_real_val, dtype=torch.float32, device=clf.device, requires_grad=False)

    size = 0

    X_neg = torch.zeros(*data_real.shape, dtype=torch.float32, device=clf.device, requires_grad=False)
    X_neg_val = torch.zeros(*data_real_val.shape, dtype=torch.float32, device=clf.device, requires_grad=False)

    if self.progress is not None:
      pbar = self.progress(total=budget)
    else:
      pbar = None

    ce = np.log(2)
    ce_val = np.log(2)

    while budget is None or size < budget:
      to_generate = min(budget - size, self.size_step)

      X_neg[size:(size + to_generate)] = torch.tensor(
        gen_neg[size:(size + to_generate)][0],
        dtype=torch.float32, device=clf.device, requires_grad=False
      )
      X_neg_val[size:(size + to_generate)] = torch.tensor(
        gen_neg_val[size:(size + to_generate)][0],
        dtype=torch.float32, device=clf.device, requires_grad=False
      )

      size += to_generate

      if self.progress is not None:
        pbar.update(to_generate)
        pbar.set_description('train = %+.3lf, test = %+.3lf' % (ce, ce_val))

      ce, ce_val = clf.adaptive_fit(X_pos, X_neg[:size], X_pos_val, X_neg_val[:size])

      if ce_val is None:
        ce = 0
        ce_val = np.log(2)
        print('premature stop due to overfit')

      if self.progress is not None:
        pbar.set_description('train = %+.3lf, test = %+.3lf' % (ce, ce_val))

      if np.abs(ce - ce_val) < self.target_diff:
        print('converged! %.3lf vs %.3lf [%d]' % (ce, ce_val, size))
        break

    else:
      return None, (ce + ce_val) / 2

    return size, (ce + ce_val) / 2