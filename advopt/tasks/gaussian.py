import numpy as np

from .meta import Task

__all__ = [
  'Gaussians',
  'AltGaussians'
]

class Gaussians(Task):
  def __init__(self, ndim=2, fixed_variance=True):
    super(Gaussians, self).__init__(ndim)

    self.fixed_variance = fixed_variance

  def name(self):
    if self.fixed_variance:
      return 'Gaussians-means-%dD' % (self._ndim,)
    else:
      return 'Gaussians-means-stds-%dD' % (self._ndim,)

  def ndim(self):
    return self._ndim

  def parameters_names(self):
    mean_names = [
      'offset_%d' % (i, )
      for i in range(self._ndim)
    ]

    std_names = [
      'std_%d' % (i,)
      for i in range(self._ndim)
    ]

    if self.fixed_variance:
      return mean_names
    else:
      return mean_names + std_names

  def search_space(self):
    mean_space = [[-3.0, 3.0]] * self._ndim
    variance_space = [[0.1, 10.0]] * self._ndim

    if self.fixed_variance:
      return mean_space
    else:
      return mean_space + variance_space

  def solution(self):
    if self.fixed_variance:
      return np.array([0.0] * self._ndim, dtype='float32')
    else:
      if self.fixed_variance:
        return np.array([0.0] * self._ndim + [1.0] * self._ndim, dtype='float32')

  def ground_truth_generator(self):
    return lambda size: np.random.normal(0, 1, size=(size, self._ndim))

  def generator(self, params):
    if callable(params):
      if self.fixed_variance:
        def gen(size):
          mean = params(size)
          return (
            np.random.normal(0, 1, size=(size, self._ndim)) + mean,
            mean,
          )

        return gen
      else:
        def gen(size):
          ps = np.asarray(params(size))
          mean = ps[:, self._ndim]
          std = ps[:, self._ndim:]

          return (
            np.random.normal(0, 1, size=(size, self._ndim)) * std + mean,
            ps,
          )

        return gen
    else:
      if self.fixed_variance:
        mean = np.asarray(params)
        return lambda size: np.random.normal(0, 1, size=(size, self._ndim)) + mean[None, :]
      else:
        mean = np.asarray(params)[:self._ndim]
        std = np.asarray(params)[self._ndim:]
        return lambda size: np.random.normal(0, 1, size=(size, self._ndim)) * std[None, :] + mean[None, :]

  def model_parameters(self):
    return dict(
      n_estimators=20,
      max_depth=3,

      n_units=128,
      grad_penalty=5e-2,
    )

class AltGaussians(Task):
  def __init__(self, ndim=2):
    super(AltGaussians, self).__init__(ndim)

  def name(self):
    return 'Gaussians'

  def parameters_names(self):
    return [r'$\theta$']

  def search_space(self):
    return [[-1.0, 4.5]]

  def solution(self):
    return np.array([0.0])

  def example_parameters(self):
    return np.array([1.0])

  def ground_truth_generator(self):
    return lambda size: np.random.normal(0, 1, size=(size, self._ndim))

  def generator(self, params):
    if callable(params):
      def gen(size):
        ps = params(size)
        mean = ps[:, 0]
        std = np.exp(mean)

        return (
          np.random.normal(0, 1, size=(size, self._ndim)) * std[:, None] + mean[:, None],
          ps
        )

      return gen

    else:
      mean = params[0]
      std = np.exp(params)

      return lambda size: np.random.normal(0, 1, size=(size, self._ndim)) * std + mean

  def transform(self, data0, params):
    mean = params[0]
    std = np.exp(params)
    return data0 * std + mean

  def model_parameters(self):
    return dict(
      n_estimators=20,
      max_depth=3,

      n_units=128,
      grad_penalty=5e-2,
    )

