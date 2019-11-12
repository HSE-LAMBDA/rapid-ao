import numpy as np

import torch
from torch.nn import functional as F

from .utils import seq, lincapacity

LOG2 = np.float32(np.log(2))

__all__ = [
  'Network',
]

class Network(object):
  def __init__(
    self, net, device='cpu',
    batch_size=32,
    min_stagnation=128,
    max_epoches=None,
    regularization=None,
    const_regularization=None,
    capacity=None,
    rho=0.99,
    stop_diff=None
  ):
    """
    :param net: an instance of torch.nn.Module
    :param device: torch device
    :param batch_size:
    :param min_stagnation: stops straining upon reaching `min_stagnation` iterations without improvement.
    :param regularization: see advopt.nn.utils module for examples.
    :param capacity: capacity function
    :param rho: coefficient for moving average used for loss estimation.
    :param stop_diff: stops training if difference between train and validation losses reaches `stop_diff`.
      This works only if validation data is supplied.
    """
    self.net = net.to(device)
    self.net_init = self.net.state_dict()

    self.opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    self.opt_init = self.opt.state_dict()

    self.device = device

    self._loss_history = list()

    self.batch_size = batch_size
    self.min_stagnation = min_stagnation
    self.max_epoches = max_epoches

    self.regularization = regularization
    self.const_regularization = const_regularization

    if capacity is None:
      self.capacity = lincapacity()
    else:
      self.capacity = capacity
    self.rho = rho

    self.stop_diff = stop_diff

    self.stochastic = hasattr(net, 'dropout') and getattr(net, 'dropout')

  def reset(self):
    self.net.load_state_dict(self.net_init)
    self.opt.load_state_dict(self.opt_init)
    self._loss_history = list()

  def soft_reset(self):
    self._loss_history = list()

  def adaptive_fit(self, X_pos, X_neg, X_pos_val=None, X_neg_val=None):
    ce_acc = torch.tensor([np.log(2)], dtype=torch.float32, device=self.device, requires_grad=False)

    if X_pos_val is not None:
      ce_acc_val = torch.tensor([np.log(2)], dtype=torch.float32, device=self.device, requires_grad=False)
    else:
      ce_acc_val = None

    min_ce = ce_acc.item()
    stagnant = 0

    stagnation_limit = max(self.min_stagnation, 2 * (X_pos.shape[0] + X_neg.shape[0]) // self.batch_size)

    iteration = 0
    if self.max_epoches is not None:
      max_iterations = (self.max_epoches * X_pos.shape[0]) // self.batch_size
    else:
      max_iterations = None

    while stagnant < stagnation_limit:
      indx_pos = torch.randint(X_pos.shape[0], size=(self.batch_size,), device=self.device)
      indx_neg = torch.randint(X_neg.shape[0], size=(self.batch_size,), device=self.device)
      X_pos_batch = X_pos[indx_pos]
      X_neg_batch = X_neg[indx_neg]

      if self.stochastic:
        _ = self.step(X_pos_batch, X_neg_batch, ce_acc=ce_acc)
        ce_batch = self.eval(X_pos_batch, X_neg_batch)
      else:
        ce_batch = self.step(X_pos_batch, X_neg_batch, ce_acc=ce_acc)

      with torch.no_grad():
        ce_acc = self.rho * ce_acc + (1 - self.rho) * ce_batch
        ce_acc_value = ce_acc.item()

        if X_pos_val is not None:
          indx_pos = torch.randint(X_pos_val.shape[0], size=(self.batch_size,), device=self.device)
          indx_neg = torch.randint(X_neg_val.shape[0], size=(self.batch_size,), device=self.device)

          X_pos_batch_val = X_pos_val[indx_pos]
          X_neg_batch_val = X_neg_val[indx_neg]

          ce_batch_val = self.eval(X_pos_batch_val, X_neg_batch_val)
          ce_acc_val = self.rho * ce_acc_val + (1 - self.rho) * ce_batch_val

        if ce_acc_value < min_ce:
          min_ce = ce_acc_value
          stagnant = 0
        else:
          stagnant += 1

      ### stop due to overfitting
      if X_pos_val is not None and self.stop_diff is not None:
        if ce_acc_val.item() - ce_acc.item() > self.stop_diff:
          print('stop due to overfit')
          return None, None

      ### exceeding max iterations
      iteration += 1
      if max_iterations is not None and iteration >= max_iterations:
        print('max iterations exceeded')
        break

    if X_pos_val is not None:
      return ce_acc.item(), ce_acc_val.item()
    else:
      return ce_acc.item()

  def fixed_fit(self, X_pos, X_neg, n_iterations=1024):
    ce_acc = torch.tensor([np.log(2)], dtype=torch.float32, device=self.device, requires_grad=False)

    for _ in range(n_iterations):
      indx_pos = torch.randint(X_pos.shape[0], size=(self.batch_size,), device=self.device)
      indx_neg = torch.randint(X_neg.shape[0], size=(self.batch_size,), device=self.device)
      X_pos_batch = X_pos[indx_pos]
      X_neg_batch = X_neg[indx_neg]

      if self.stochastic:
        _ = self.step(X_pos_batch, X_neg_batch, ce_acc=ce_acc)
        ce_batch = self.eval(X_pos_batch, X_neg_batch)
      else:
        ce_batch = self.step(X_pos_batch, X_neg_batch, ce_acc=ce_acc)

      with torch.no_grad():
        ce_acc = self.rho * ce_acc + (1 - self.rho) * ce_batch

    return ce_acc

  def alt_fit(self, X_pos, X_neg, X_pos_val=None, X_neg_val=None):
    X_pos = torch.tensor(X_pos, dtype=torch.float32, device=self.device, requires_grad=False)
    X_neg = torch.tensor(X_neg, dtype=torch.float32, device=self.device, requires_grad=False)

    if X_pos_val is not None:
      X_pos_val = torch.tensor(X_pos_val, dtype=torch.float32, device=self.device, requires_grad=False)
      X_neg_val = torch.tensor(X_neg_val, dtype=torch.float32, device=self.device, requires_grad=False)

    return self.adaptive_fit(X_pos, X_neg, X_pos_val=X_pos_val, X_neg_val=X_neg_val)

  def fit(self, X, y, reset=True):
    if reset:
      self.reset()

    X_pos = X[y > 0.5]
    X_neg = X[y < 0.5]
    return self.alt_fit(X_pos, X_neg)

  def loss_history(self):
    return np.array(self._loss_history)

  def predict_proba(self, X, flatten=False):
    result = np.ndarray(shape=(X.shape[0], ), dtype='float32')

    with torch.no_grad():
      for indx in seq(X.shape[0], self.batch_size):
        X_batch = torch.tensor(X[indx], dtype=torch.float32, device=self.device)
        logits = self.net(X_batch)
        proba = torch.sigmoid(logits)
        result[indx] = proba.detach().cpu().numpy()

    if flatten:
      return result
    else:
      return np.vstack([1 - result, result]).T

  def predict_grid(self, space, steps=20):
    xs = tuple([
      np.linspace(lower_x, upper_x, num=steps)
      for lower_x, upper_x in space
    ])

    grids = np.meshgrid(*xs)
    X = np.vstack([grid.reshape(-1) for grid in grids]).T
    proba = self.predict_proba(X, flatten=True)
    proba = proba.reshape((steps, ) * len(space))

    return xs, proba

  def eval(self, X_pos, X_neg):
    with torch.no_grad():
      proba_pos = self.net(X_pos)
      proba_neg = self.net(X_neg)

      ce = 0.5 * (
        torch.mean(F.softplus(-proba_pos)) +
        torch.mean(F.softplus(proba_neg))
      )

    return ce

  def step(self, X_pos, X_neg, ce_acc=None):
    """
    :param ce_acc: current estimate of cross-entropy loss (on the whole dataset),
      required for the regularization (if present).
    :return: cross-entropy measured on `X_pos` and `X_neg`.
    """
    self.opt.zero_grad()

    if ce_acc is not None:
      with torch.no_grad():
        coef = self.capacity(ce_acc / LOG2)
    else:
      coef = None

    try:
      proba_pos = self.net(X_pos, coef)
      proba_neg = self.net(X_neg, coef)
    except TypeError:
      proba_pos = self.net(X_pos)
      proba_neg = self.net(X_neg)

    ce = 0.5 * (
      torch.mean(F.softplus(-proba_pos)) +
      torch.mean(F.softplus(proba_neg))
    )

    if self.regularization is not None and coef is not None:
      reg = self.regularization(self.net, X_pos, X_neg, proba_pos, proba_neg)
      loss = ce + coef * reg
    else:
      loss = ce

    if self.const_regularization is not None:
      const_reg = self.const_regularization(self.net, X_pos, X_neg, proba_pos, proba_neg)
      loss = loss + const_reg

    loss.backward()
    self.opt.step()

    return ce

