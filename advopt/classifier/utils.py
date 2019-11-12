import numpy as np

__all__ = [
  'linspace', 'logspace',
]

def linspace(num):
  return np.linspace(np.log(2), 0.0, num=num)

def logspace(num, coef=5):
  a = np.exp(-coef * np.linspace(0, 1, num=num))
  return np.log(2) * (a - np.exp(-coef)) / (1 - np.exp(-coef))
