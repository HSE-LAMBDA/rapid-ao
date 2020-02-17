import numpy as np

from .meta import metric, cached_generators
from .search import search
from .utils import combine

__all__ = [
  'adjusted',
  'prob_criterion',
  'diff_criterion',
  'semiprob_criterion',
  'logloss'
]

def logloss(y, p, eps=1e-6):
  return -(
    y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)
  )

class diff_criterion(object):
  def __init__(self, tolerance):
    self.tolerance = tolerance

  def __call__(self, fs_train, fs_val):
    return np.mean(fs_train) - np.mean(fs_val) - self.tolerance

class prob_criterion(object):
  """
  This criterion treats estimate of average losses as a normally distributed random variables.
  The assumption of normality is valid due to relatively high number of samples.

  Returns `confidence - P(|L_train - L_val| < tolerance)`,
  where L_train, L_val are average losses on train and test samples,
  assumed to be normally distributed.
  """
  def __init__(self, tolerance, confidence):
    self.tolerance = tolerance
    self.confidence = confidence

  def __call__(self, fs_train, fs_val):
    if len(fs_train) < 1:
      return 1

    tolerance, confidence = self.tolerance, self.confidence

    from scipy.stats import norm
    mean_train, std_train = np.mean(fs_train), np.std(fs_train, ddof=1) / np.sqrt(fs_train.shape[0])
    mean_val, std_val = np.mean(fs_val), np.std(fs_val, ddof=1) / np.sqrt(fs_val.shape[0])

    mean = mean_train - mean_val
    std = np.sqrt(std_train ** 2 + std_val ** 2)

    prob_interval = norm.cdf(tolerance / 2, mean, std) - norm.cdf(-tolerance / 2, mean, std)

    return confidence - prob_interval

class semiprob_criterion(object):
  """
  This criterion treats estimate of average losses as a normally distributed random variables.
  The assumption of normality is valid due to relatively high number of samples.

  Returns `max(delta / std_val, delta / std_train) - relative_tolerance`,
  where std_val, std_train are standard deviations of train/validation losses,
  delta is difference between average train and validation losses.
  """
  def __init__(self, relative_tolerance=0.5):
    self.relative_tolerance = relative_tolerance

  def __call__(self, fs_train, fs_val):
    if len(fs_train) < 1:
      return 1

    mean_train, std_train = np.mean(fs_train), np.std(fs_train, ddof=1) / np.sqrt(fs_train.shape[0])
    mean_val, std_val = np.mean(fs_val), np.std(fs_val, ddof=1) / np.sqrt(fs_val.shape[0])

    delta = mean_train - mean_val
    return max(delta / std_train, delta / std_val) - self.relative_tolerance

def fit(clf, X_pos_train, X_neg_train, X_pos_val, X_neg_val, average=True):
  if X_pos_train.shape[0] == 0 or X_neg_train.shape[0] == 0:
    return np.log(2), 0

  X_train, y_train = combine(X_pos_train, X_neg_train)

  if X_pos_val is not None:
    X_val, y_val = combine(X_pos_val, X_neg_val)
  else:
    X_val, y_val = None, None

  clf.fit(X_train, y_train)

  proba_train = clf.predict_proba(X_train)[:, 1]
  jsd_train = np.log(2) - logloss(y_train, proba_train)

  if X_val is not None:
    proba_val = clf.predict_proba(X_val)[:, 1]
    jsd_val = np.log(2) - logloss(y_val, proba_val)
  else:
    jsd_val = None

  if average:
    return np.mean(jsd_train), np.mean(jsd_val) if jsd_val is not None else None
  else:
    return jsd_train, jsd_val if jsd_val is not None else None



class adjusted(metric):
  def __init__(
    self, criterion=diff_criterion(1e-2),
    xtol=128, x0=None, diff_stop=np.log(2) / 2,
    search_method='bisect', verbose=False
  ):
    super(adjusted, self).__init__()

    self.criterion = criterion
    self.xtol = xtol
    self.x0 = x0
    self.search_method = search_method
    self.diff_stop = diff_stop
    self.verbose = verbose

  def __call__(self, clf, gen_pos, gen_neg, gen_pos_val=None, gen_neg_val=None, budget=None):
    gen_pos, gen_pos_val = cached_generators(gen_pos, gen_pos_val)
    gen_neg, gen_neg_val = cached_generators(gen_neg, gen_neg_val)

    def m(size):
      size = int(size)

      X_pos_train = gen_pos.samples(size)
      X_neg_train = gen_neg.samples(size)
      X_pos_val = gen_pos_val.samples(size)
      X_neg_val = gen_neg_val.samples(size)

      jsd_train, jsd_val = fit(
        clf, X_pos_train, X_neg_train, X_pos_val, X_neg_val,
        average=False
      )

      return jsd_train, jsd_val

    def target(size):
      jsd_train, jsd_val = m(size)
      return self.criterion(jsd_train, jsd_val)

    size0 = search(
      target,
      xtol=self.xtol,
      x0=self.x0,
      method=self.search_method,
      verbose=self.verbose,
      limit=budget
    )

    if size0 is None:
      return None, None

    jsd_train, jsd_val = m(size0)
    return size0, (np.mean(jsd_train) + np.mean(jsd_val)) / 2
