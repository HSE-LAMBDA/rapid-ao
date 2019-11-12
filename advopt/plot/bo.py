import numpy as np

__all__ = [
  'plot_bo'
]

def plot_bo(bo):
  import matplotlib.pyplot as plt
  if len(bo.models) == 0:
    return

  model = bo.models[-1]

  a, b = bo.space.bounds[0]
  xs = np.linspace(0, 1, num=100)

  mean, std = model.predict(xs.reshape(-1, 1), return_std=True)

  X = np.array(bo.Xi)
  y = np.array(bo.yi)
  plt.figure()
  plt.scatter(X[:, 0], y, color='green')
  plt.fill_between(np.linspace(a, b, num=100), mean - std, mean + std, alpha=0.2, color='blue')
  plt.plot(np.linspace(a, b, num=100), mean, color='blue')
  plt.show()