import numpy as np
import torch
from torch import autograd

__all__ = [
  'lincapacity', 'logcapacity', 'constcapacity',
  'l2_reg', 'grad_reg', 'r1_reg',

  'seq',
]

def cummin(arr, x0=np.log(2)):
  c = x0
  result = np.zeros_like(arr)

  for i, x in enumerate(arr):
    if x < c:
      result[i] = x
      c = x
    else:
      result[i] = c

  return result

def seq(n_samples, batch_size=None):
  indx = np.arange(n_samples)

  if batch_size is None:
    for i in indx:
      yield i

  else:
    n_batches = n_samples // batch_size + (1 if n_samples % batch_size != 0 else 0)

    for i in range(n_batches):
      i_from = i * batch_size
      i_to = i_from + batch_size
      yield indx[i_from:i_to]

class constcapacity(object):
  def __init__(self, capacity, device=None):
    self.capacity = torch.tensor(capacity, device=device, dtype=torch.float32)

  def __call__(self, t):
    return self.capacity

class lincapacity(object):
  def   __init__(self, coef=None):
    self.coef = coef

  def __call__(self, t):
    if self.coef is None:
      return torch.clamp(1 - t, 0, 1)
    else:
      return torch.clamp(self.coef - self.coef * t, 0, 1)

class logcapacity(object):
  def __init__(self, coef=None):
    self.coef = coef

  def __call__(self, t):
    if self.coef is None:
      return -torch.log(torch.clamp_min(t, 1e-2))
    else:
      return -self.coef * torch.log(torch.clamp_min(t, 1e-2))


def dropout(X, p):
  if p is not None:
    p = torch.clamp_max(p, 0.9)
    drop = torch.rand(X.shape[:2], device=X.device, dtype=torch.float32, requires_grad=False)

    drop = torch.where(drop < p, torch.zeros_like(drop), torch.ones_like(drop))

    left = torch.clamp_min(torch.sum(drop, dim=1), 1)
    normalization = X.shape[1] / left

    broadcast = (slice(None, None, None), slice(None, None, None)) + (None,) * (len(X.shape) - 2)

    return X * (drop * normalization[:, None])[broadcast]
  else:
    return X

class l2_reg(object):
  def __init__(self, coef=1e-3):
    self.coef = coef

  def __call__(self, net, X_pos, X_neg, p_pos, p_neg):
    reg = None
    for param in net.parameters():
      if param.ndimension() == 2 or param.ndimension() == 4:
        if reg is None:
          reg = torch.sum(param ** 2)
        else:
          reg = reg + torch.sum(param ** 2)

    return self.coef * reg

class r1_reg(object):
  def __init__(self, coef=1e-3):
    self.coef = coef

  def __call__(self, net, X_pos, X_neg, p_pos, p_neg):
    grads = autograd.grad(
      torch.mean(p_pos),
      net.parameters(),
      create_graph=True,
      retain_graph=True
    )

    return self.coef * sum([torch.mean(grad ** 2) for grad in grads])

class grad_reg(object):
  def __init__(self, coef=1e-3):
    self.coef = coef

  def __call__(self, net, X_pos, X_neg, p_pos, p_neg):
    grad, = autograd.grad(
      torch.mean(p_pos),
      X_pos,
      create_graph=True,
      retain_graph=True
    )

    return self.coef * torch.mean(grad ** 2)