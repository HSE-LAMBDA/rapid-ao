import numpy as np

__all__ = [
  'linspace', 'logspace',
]

def linspace(num, x0=np.log(2)):
  return np.linspace(x0, 0, num=num)

def logspace(num, x0=np.log(2)):
  a = np.linspace(0, 1, num=num + 1)[1:]
  return x0 * np.log(a) / np.log(a[0])
