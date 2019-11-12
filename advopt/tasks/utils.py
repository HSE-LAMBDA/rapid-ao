import numpy as np

__all__ = [
  'rot_matrix',
  'gbdt_models',
  'nn_models'
]

def gbdt_models(ndim, n_estimators=100, max_depth=5, x0_lin=np.log(2), x0_log=np.log(2)):
  from sklearn.ensemble import GradientBoostingClassifier
  from ..classifier import AdaptiveBoosting, logspace, linspace

  from catboost import CatBoostClassifier
  catboost_common = dict(
    thread_count=8,
    od_pval=0,
    learning_rate=0.1,
    verbose=False
  )

  return {
    'JSD-GBDT' : lambda seed: GradientBoostingClassifier(
      n_estimators=n_estimators, max_depth=max_depth,
      random_state=seed
    ),

    'log-pJSD-GBDT' : lambda seed: AdaptiveBoosting(
      GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed
      ),
      thresholds=logspace(n_estimators, x0=x0_log)
    ),

    'lin-pJSD-GBDT': lambda seed: AdaptiveBoosting(
      GradientBoostingClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed
      ),
      thresholds=linspace(n_estimators, x0=x0_lin)
    ),

    'JSD-cat': lambda seed: CatBoostClassifier(
      loss_function='CrossEntropy', n_estimators=n_estimators, max_depth=max_depth,
      random_state=seed, **catboost_common
    ),

    'log-pJSD-cat': lambda seed: AdaptiveBoosting(
      CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, **catboost_common
      ),
      thresholds=logspace(n_estimators, x0=x0_log)
    ),

    'lin-pJSD-cat': lambda seed: AdaptiveBoosting(
      CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=n_estimators, max_depth=max_depth,
        random_state=seed, **catboost_common
      ),
      thresholds=linspace(n_estimators, x0=x0_lin)
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
      thresholds=logspace(n_estimators, x0=x0_log)
    ),

    'lin-pJSD-cat-20': lambda seed: AdaptiveBoosting(
      CatBoostClassifier(
        loss_function='CrossEntropy', n_estimators=20, max_depth=2,
        random_state=seed, **catboost_common
      ),
      thresholds=linspace(n_estimators, x0=x0_lin)
    ),
  }

def nn_models(ndim, n_units=128, grad_penalty=1e-2, reg_penalty=1e-3):
  from ..classifier import Dense, Network
  from ..classifier import grad_reg, l2_reg
  from ..classifier.nn.utils import lincapacity, logcapacity

  models = {
    'JSD-NN': lambda device: Network(
      net=Dense(ndim=ndim, n0=n_units),
      device=device,
      regularization=None
    ),

    'lin-pJSD-grad': lambda device: Network(
      net=Dense(ndim=ndim, n0=n_units),
      device=device,
      regularization=grad_reg(grad_penalty),
      capacity=lincapacity()
    ),

    'log-pJSD-grad' : lambda device: Network(
      net=Dense(ndim=ndim, n0=n_units),
      device=device,
      regularization=grad_reg(grad_penalty),
      capacity=logcapacity()
    ),

    'lin-pJSD-reg': lambda device: Network(
      net=Dense(ndim=ndim, n0=n_units),
      device=device,
      regularization=l2_reg(reg_penalty),
      capacity=lincapacity()
    ),

    'log-pJSD-reg': lambda device: Network(
      net=Dense(ndim=ndim, n0=n_units),
      device=device,
      regularization=l2_reg(reg_penalty),
      capacity=logcapacity()
    ),
  }

  return models

def rot_matrix(angle):
  return np.array([
    [np.cos(angle), -np.sin(angle)],
    [np.sin(angle), np.cos(angle)],
  ])