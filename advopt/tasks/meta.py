import numpy as np

__all__ = [
  'Task'
]

class Task(object):
  def __init__(self, ndim=None, seed=None):
    self._ndim = ndim
    self.rng = np.random.RandomState(seed)

  def set_seed(self, seed=None):
    self.rng = np.random.RandomState(seed)

  def name(self):
    return self.__class__.__name__

  def ndim(self):
    if self._ndim is None:
      self._ndim = self.ground_truth_generator()(1).shape[0]

    return self._ndim

  def search_space(self):
    raise NotImplementedError()

  def parameters_names(self):
    return [
      'param_%d' % (i, )
      for i in range(len(self.search_space()))
    ]

  def solution(self):
    raise NotImplementedError

  def example_parameters(self):
    state = np.random.get_state()
    np.random.seed(1234444)
    ss = np.array(self.search_space())
    u = np.random.uniform(size=len(ss))

    result = u * (ss[:, 1] - ss[:, 0]) + ss[:, 0]

    np.random.set_state(state)
    return result

  def ground_truth_generator(self):
    raise NotImplementedError()

  def generator(self, params):
    raise NotImplementedError()

  def transform(self, data0, params):
    """
    This function transforms sample `data0` from the ground-truth generator
    into sample from a generator with parameters `params`.

    Useful only for synthetic examples.
    """
    raise NotImplementedError()

  def is_synthetic(self):
    try:
      self.transform(None, None)
    except NotImplementedError:
      return False
    except:
      return True

  def model_parameters(self):
    return dict()

  def models(self, seed=None):
    from .utils import nn_models, gbdt_models
    from ..meta import apply_with_kwargs

    parameters = self.model_parameters()
    parameters['ndim'] = self.ndim()
    if seed is not None:
      parameters['seed'] = seed
      parameters['random_state'] = seed

    ms = dict()

    ms.update(
      apply_with_kwargs(nn_models, **parameters)
    )
    ms.update(
      apply_with_kwargs(gbdt_models, **parameters)
    )

    return ms

  def optimizers(self):
    from skopt.learning import GaussianProcessRegressor
    from skopt.learning.gaussian_process.kernels import Matern

    from ..opt import bayesian_optimization, random_search

    return {
      'BO' : bayesian_optimization(
        base_estimator=GaussianProcessRegressor(
            kernel=Matern(length_scale=1, length_scale_bounds=(1e-3, 1e+3)),
            alpha=1e-4
        ),
        n_initial_points=5
      ),

      'RS' : random_search()
    }