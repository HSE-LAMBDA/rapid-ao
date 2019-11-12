import numpy as np
from advopt.target import logloss


__all__ = [
  'AdaptiveBoosting',
]

class AdaptiveBoosting(object):
  def __init__(self, clf, thresholds):
    self.clf = clf
    self.thresholds = thresholds
    self.capacity = 0

  def fit(self, X, y):
    self.clf.fit(X, y)
    proba_iter = self.clf.staged_predict_proba(X)

    self.capacity = 0

    for i, proba in enumerate(proba_iter):
      self.capacity += 1
      jsd = np.log(2) - np.mean(logloss(y, proba[:, 1]))

      if jsd > self.thresholds[i]:
        break

  def predict_proba(self, X):
    proba_iter = self.clf.staged_predict_proba(X)

    for _ in range(self.capacity - 1):
      next(proba_iter)

    return next(proba_iter)