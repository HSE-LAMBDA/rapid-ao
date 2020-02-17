import numpy as np

__all__ = [
  'combine'
]


def combine(X_pos, X_neg):
  X = np.concatenate([X_pos, X_neg], axis=0)
  y = np.concatenate([
    np.ones(X_pos.shape[0], dtype='float32'),
    np.zeros(X_neg.shape[0], dtype='float32'),
  ], axis=0)

  return X, y