import numpy as np

from .meta import cached_generators, metric
from .utils import combine
from .adjusted import logloss


class fixed_size(metric):
  def __init__(self, size):
    super(fixed_size, self).__init__()
    self.size = size

  def __call__(self, clf, gen_pos, gen_neg, gen_pos_val=None, gen_neg_val=None, budget=None):
    gen_pos, gen_pos_val = cached_generators(gen_pos, gen_pos_val)
    gen_neg, gen_neg_val = cached_generators(gen_neg, gen_neg_val)

    X_pos = gen_pos.samples(self.size)
    X_neg = gen_neg.samples(self.size)

    X_pos_val = gen_pos_val.samples(self.size)
    X_neg_val = gen_neg_val.samples(self.size)

    X, y = combine(X_pos, X_neg)
    X_val, y_val = combine(X_pos_val, X_neg_val)

    clf.fit(X, y)

    proba = clf.predict_proba(X)[:, 1]
    proba_val = clf.predict_proba(X_val)[:, 1]

    jsd = np.log(2) - np.mean(logloss(y, proba))
    jsd_val = np.log(2) - np.mean(logloss(y_val, proba_val))

    return jsd, jsd_val


