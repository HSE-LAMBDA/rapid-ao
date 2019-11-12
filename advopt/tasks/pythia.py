import numpy as np
import pythiamill as pm

from .meta import Task

__all__ = [
  'PythiaTuneMC',
  'PythiaTracker'
]

fixed_options = [
  ### seeting default parameters to Monash values
  "Tune:ee = 7",
  "Beams:idA = 11",
  "Beams:idB = -11",
  "Beams:eCM = 91.2",
  "WeakSingleBoson:ffbar2gmZ = on",
  "23:onMode = off",
  "23:onIfMatch = 1 -1",
  "23:onIfMatch = 2 -2",
  "23:onIfMatch = 3 -3",
  "23:onIfMatch = 4 -4",
  "23:onIfMatch = 5 -5",
]

param_names = [
  "TimeShower:alphaSvalue",
  # "TimeShower:pTmin",
  # "TimeShower:pTminChgQ",

  # "StringPT:sigma",
  # "StringZ:bLund",
  # "StringZ:aExtraSQuark",
  # "StringZ:aExtraDiquark",
  # "StringZ:rFactC",
  # "StringZ:rFactB",

  "StringFlav:probStoUD",
  "StringFlav:probQQtoQ",
  "StringFlav:probSQtoQQ",
  "StringFlav:probQQ1toQQ0",
  "StringFlav:mesonUDvector",
  "StringFlav:mesonSvector",
  "StringFlav:mesonCvector",
  "StringFlav:mesonBvector",
  "StringFlav:etaSup",
  "StringFlav:etaPrimeSup",
  "StringFlav:decupletSup"
]

space = [
  (0.06, 0.25),
  # (0.1, 2.0),
  # (0.1, 2.0),

  # (0.2, 1.0),
  # (0.0, 1.0),
  # (0.0, 2.0),
  # (0.0, 2.0),
  # (0.0, 2.0),
  # (0.0, 2.0),

  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 1.0),
  (0.0, 3.0),
  (0.0, 3.0),
  (0.0, 3.0),
  (0.0, 3.0)
]

monash = np.array([
  0.1365,
  # 0.5,
  # 0.5,

  # 0.98,
  # 0.335,
  # 0,
  # 0.97,
  # 1.32,
  # 0.885,

  0.217,
  0.081,
  0.915,
  0.0275,
  0.6,
  0.12,
  1,
  0.5,
  0.55,
  0.88,
  2.2
])

class PythiaTuneMC(Task):
  def __init__(self, n_params=None, n_jobs=None, seed=None):
    import pythiamill as pm
    self.detector = pm.utils.TuneMCDetector()

    if n_params is None:
      n_params = len(param_names)

    assert n_params <= len(param_names)
    self.n_params = n_params
    self.n_jobs = n_jobs

    self.seed = seed

    super(PythiaTuneMC, self).__init__(ndim=self.detector.event_size(), seed=seed)

  def set_seed(self, seed=None):
    self.seed = seed

  def name(self):
    return 'Pythia-%d' % (self.n_params, )

  def parameters_names(self):
    return param_names[:self.n_params]

  def search_space(self):
    return space[:self.n_params]

  def solution(self):
    return monash[:self.n_params]

  def ground_truth_generator(self):
    import pythiamill as pm
    params = monash[:self.n_params]


    return pm.CachedPythiaMill(
      detector_factory=self.detector,
      options=pm.config.please_be_quiet + fixed_options + [
        '%s = %lf' % (k, v)
        for k, v in zip(param_names, params)
      ],
      batch_size=32,
      n_workers=self.n_jobs,
      seed=self.seed,
      log='./pythia-log'
    )

  def generator(self, params):
    import pythiamill as pm

    return pm.CachedPythiaMill(
      detector_factory=self.detector,
      options=pm.config.please_be_quiet + fixed_options + [
        '%s = %lf' % (k, v)
        for k, v in zip(param_names, params)
      ],
      batch_size=32,
      n_workers=self.n_jobs,
      seed=self.seed,
      log='./pythia-log'
    )

  def models(self, seed=None):
    from sklearn.ensemble import GradientBoostingClassifier
    from ..classifier import AdaptiveBoosting
    from ..classifier import linspace, logspace

    from catboost import CatBoostClassifier

    catboost_common = dict(
      thread_count=8,
      od_pval=0,
      learning_rate=0.1,
      verbose=False
    )

    models = {
      'JSD-GBDT' : lambda seed: GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        random_state=seed
      ),

      'log-pJSD-GBDT' : lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=100, max_depth=3,
          random_state=seed
        ),
        thresholds=logspace(100, x0=np.log(2) / 4)
      ),

      'lin-pJSD-GBDT': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=100, max_depth=3,
          random_state=seed
        ),
        thresholds=linspace(100, x0=np.log(2) / 4)
      ),

      'JSD-GBDT-20': lambda seed: GradientBoostingClassifier(
        n_estimators=20, max_depth=2,
        random_state=seed
      ),

      'log-pJSD-GBDT-20': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=20, max_depth=2,
          random_state=seed
        ),
        thresholds=logspace(20, x0=np.log(2) / 4)
      ),

      'lin-pJSD-GBDT-20': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=20, max_depth=2,
          random_state=seed
        ),
        thresholds=linspace(20, x0=np.log(2) / 4)
      ),

      'JSD-GBDT-10': lambda seed: GradientBoostingClassifier(
        n_estimators=10, max_depth=1,
        random_state=seed
      ),

      'log-pJSD-GBDT-10': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=10, max_depth=1,
          random_state=seed
        ),
        thresholds=logspace(10, x0=np.log(2) / 4)
      ),

      'lin-pJSD-GBDT-10': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=10, max_depth=1,
          random_state=seed
        ),
        thresholds=linspace(10, x0=np.log(2) / 4)
      ),

      'JSD-cat': lambda seed: CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=100, max_depth=3,
        random_state=seed, **catboost_common
      ),

      'log-pJSD-cat': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=100, max_depth=3,
          random_state=seed, **catboost_common
        ),
        thresholds=logspace(100, x0=np.log(2) / 4)
      ),

      'lin-pJSD-cat': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=100, max_depth=3,
          random_state=seed, **catboost_common
        ),
        thresholds=linspace(100, x0=np.log(2) / 4)
      ),

      'JSD-cat-20': lambda seed: CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=20, max_depth=2,
        random_state=seed, **catboost_common
      ),

      'log-pJSD-cat-20': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=20, max_depth=2,
          random_state=seed, **catboost_common
        ),
        thresholds=logspace(20, x0=np.log(2) / 4)
      ),

      'lin-pJSD-cat-20': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=20, max_depth=2,
          random_state=seed, **catboost_common
        ),
        thresholds=linspace(20, x0=np.log(2) / 4)
      ),

      'JSD-cat-10': lambda seed: CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=10, max_depth=2,
        random_state=seed, **catboost_common
      ),

      'log-pJSD-cat-10': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=10, max_depth=2,
          random_state=seed, **catboost_common
        ),
        thresholds=logspace(20, x0=np.log(2) / 4)
      ),

      'lin-pJSD-cat-10': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=10, max_depth=2,
          random_state=seed, **catboost_common
        ),
        thresholds=linspace(20, x0=np.log(2) / 4)
      ),
    }

    return models

class PythiaMixtureSampler(object):
  def __init__(self, shape,  mill : pm.ParametrizedPythiaMill, param_distribution):
    self.shape = shape
    self.mill = mill

    if callable(param_distribution):
      self.param_distribution = param_distribution
    else:
      self.param_distribution = lambda size: np.repeat(param_distribution[None, :], repeats=size, axis=0)

  def __call__(self, size):
    params = self.param_distribution(size)

    for param in params:
      self.mill.request(param)

    retrieved_params, samples = zip(*[
      self.mill.retrieve()
      for _ in params
    ])

    samples = np.concatenate(samples, axis=0)
    retrieved_params = np.concatenate(retrieved_params, axis=0)

    ### kind of normalization
    np.log1p(samples, out=samples)

    samples = samples.reshape((samples.shape[0], ) + self.shape)

    return samples, retrieved_params

class PythiaTracker(Task):
  def __init__(self, n_jobs=None, seed=None, flatten=False):
    self.shape = (1, 32, 32)
    self._ndim = self.shape

    if flatten:
      self.target_shape = (1024, )
    else:
      self.target_shape = self.shape

    self.detector = pm.utils.SphericalTracker(
      is_binary=False, photon_detection=True,
      pseudorapidity_steps=self.shape[1], phi_steps=self.shape[2],
      n_layers=self.shape[0], R_min=1, R_max=1,
    )
    self.n_jobs = n_jobs
    self.seed = seed

    self.flatten = flatten

    ### pythia-mill might not be properly serializable.
    self._mill = None

    self.options = pm.config.monash + pm.config.please_be_quiet + [
      '%s = %lf' % (k, v)
      for k, v in zip(param_names, monash)
    ]

    super(PythiaTracker, self).__init__(ndim=self.shape, seed=seed)

  def set_seed(self, seed=None):
    self.seed = seed

    if self._mill is not None:
      self._mill.shutdown()
      self._mill = None

  def name(self):
    return 'PythiaTracker'

  def parameters_names(self):
    return ['offset']

  def search_space(self):
    return [
      (-1.0, 1.0),
      (-1.0, 1.0),
      (-1.0, 1.0)
    ]

  def solution(self):
    return np.array([0.0, 0.0, 0.0], dtype='float32')

  def mill(self):
    if self._mill is None:
      self._mill = pm.ParametrizedPythiaMill(
        detector_factory=self.detector,
        options=self.options,
        batch_size=1,
        n_workers=self.n_jobs,
        seed=self.seed
      )

    return self._mill

  def ground_truth_generator(self):
    return PythiaMixtureSampler(self.target_shape, self.mill(), np.array([0.0], dtype='float64'))

  def generator(self, params):
    return PythiaMixtureSampler(self.target_shape, self.mill(), params)

  def models(self, seed=None):
    from ..classifier import Network, Dense
    from ..classifier import grad_reg, l2_reg, r1_reg
    from ..classifier.nn.utils import lincapacity, logcapacity

    from sklearn.ensemble import GradientBoostingClassifier
    from ..classifier import AdaptiveBoosting
    from ..classifier import linspace, logspace

    from catboost import CatBoostClassifier

    models = {
      'pJSD-dropout' : lambda device: Network(
        Dense(self.ndim(), dropout=True),
        device=device, min_stagnation=1024,
        regularization=None, capacity=lincapacity(3),
        const_regularization=r1_reg(1),
        stop_diff=1e-1
      ),

      'pJSD-l2' : lambda device : Network(
        Dense(self.ndim(), dropout=False),
        device=device, min_stagnation=1024,
        regularization=l2_reg(1e-2), capacity=logcapacity(10),
        const_regularization=r1_reg(1),
        stop_diff=1e-1
      ),

      'JSD-NN' : lambda device: Network(
        Dense(self.ndim(), dropout=False),
        device=device, min_stagnation=1024,
        stop_diff=1e-1,
        regularization=None,
        const_regularization=r1_reg(1)
      ),

      'JSD-GBDT': lambda seed: GradientBoostingClassifier(
        n_estimators=100, max_depth=3,
        random_state=seed
      ),

      'log-pJSD-GBDT': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=100, max_depth=3,
          random_state=seed
        ),
        thresholds=logspace(100)
      ),

      'lin-pJSD-GBDT': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=100, max_depth=3,
          random_state=seed
        ),
        thresholds=linspace(100)
      ),

      'JSD-GBDT-20': lambda seed: GradientBoostingClassifier(
        n_estimators=20, max_depth=2,
        random_state=seed
      ),

      'log-pJSD-GBDT-20': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=20, max_depth=2,
          random_state=seed
        ),
        thresholds=logspace(20)
      ),

      'lin-pJSD-GBDT-20': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=20, max_depth=2,
          random_state=seed
        ),
        thresholds=linspace(20)
      ),

      'JSD-GBDT-10': lambda seed: GradientBoostingClassifier(
        n_estimators=10, max_depth=1,
        random_state=seed
      ),

      'log-pJSD-GBDT-10': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=10, max_depth=1,
          random_state=seed
        ),
        thresholds=logspace(10)
      ),

      'lin-pJSD-GBDT-10': lambda seed: AdaptiveBoosting(
        GradientBoostingClassifier(
          n_estimators=10, max_depth=1,
          random_state=seed
        ),
        thresholds=linspace(10)
      ),

      'JSD-cat': lambda seed: CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=100, max_depth=3,
        random_state=seed, thread_count=8, od_pval=0
      ),

      'log-pJSD-cat': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=100, max_depth=3,
          random_state=seed, thread_count=8, od_pval=0
        ),
        thresholds=logspace(100)
      ),

      'lin-pJSD-cat': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=100, max_depth=3,
          random_state=seed, thread_count=8, od_pval=0
        ),
        thresholds=linspace(100)
      ),

      'JSD-cat-20': lambda seed: CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=20, max_depth=2,
        random_state=seed, thread_count=8, od_pval=0
      ),

      'log-pJSD-cat-20': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=20, max_depth=2,
          random_state=seed, thread_count=8, od_pval=0
        ),
        thresholds=logspace(20)
      ),

      'lin-pJSD-cat-20': lambda seed: AdaptiveBoosting(
        CatBoostClassifier(
          loss_function='CrossEntropy', n_estimators=20, max_depth=2,
          random_state=seed, thread_count=8, od_pval=0
        ),
        thresholds=linspace(20)
      ),
    }



    return models

  def optimizers(self):
    from skopt.learning import GaussianProcessRegressor
    from skopt.learning.gaussian_process.kernels import Matern

    from ..opt import bayesian_optimization, variational_optimization

    return {
      'BO' : bayesian_optimization(
        base_estimator=GaussianProcessRegressor(
          kernel=Matern(length_scale=1, length_scale_bounds=(1e-2, 1e+2)),
          alpha=1e-4
        ),
        n_initial_points=5
      ),

      'VO' : variational_optimization(
        sigma=None, sigma0=0.25, sigma_min=0.1,
        device='cpu', learning_rate=0.05
      )
    }