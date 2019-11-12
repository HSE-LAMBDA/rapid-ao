import numpy as np

import torch
from torch import nn

from advopt.classifier.nn.utils import dropout

__all__ = [
  'Dense'
]

class Dense(nn.Module):
  def __init__(self, ndim, hidden_units=32, dropout=False):
    super(Dense, self).__init__()

    if isinstance(ndim, int):
      self.inputs = ndim
    else:
      self.inputs = np.prod(ndim, dtype='int')

    self.hidden_units = hidden_units
    self.h1 = nn.Linear(self.inputs, hidden_units)
    self.activation = nn.ReLU()

    self.h2 = nn.Linear(hidden_units, 1)

    self.dropout = dropout

  def forward(self, X, p=None):
    if p is not None and self.dropout:
      drop = lambda X: dropout(X, p)
    else:
      drop = lambda X: X

    result = X
    result = drop(
      torch.flatten(result, 1)
    )
    result = drop(
      self.activation(self.h1(result))
    )

    result = self.h2(result)

    return torch.squeeze(result, dim=1)