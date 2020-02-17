import numpy as np

__all__ = [
  'cached_generator',
  'cached_generators',
  'metric'
]

class cached_generator(object):
  def __init__(self, gen):
    self.gen = gen
    self._data = None

  def __getitem__(self, item):
    if isinstance(item, slice):
      self(item.stop)
    else:
      self._cache(item)

    return tuple(x[item] for x in self._data)

  def samples(self, size):
    """
    Some generators are mixture generators that also
    return parameters along with the samples.
    This method always returns only samples.
    """
    return self(size)[0]

  def _cache(self, size):
    return tuple(x[:size] for x in self._data)

  def __call__(self, size):
    if self._data is None:
      result = self.gen(size)

      if isinstance(result, tuple):
        self._data = result
      else:
        self._data = (result, )

    while self._data[0].shape[0] < size:
      n = size - self._data[0].shape[0]

      result = self.gen(n)
      if not isinstance(result, tuple):
        result = (result, )

      self._data = tuple(
        np.concatenate([x, r], axis=0)
        for x, r in zip(self._data, result)
      )

    return self._cache(size)

def cached_generators(gen, gen_val=None):
  if gen_val is None:
    if isinstance(gen, cached_generator):
      raise ValueError('Can not use cached generator for training and validation!')

    gen_val = gen

  gen = gen if isinstance(gen, cached_generator) else cached_generator(gen)
  gen_val = gen_val if isinstance(gen_val, cached_generator) else cached_generator(gen_val)
  return gen, gen_val

### well, it is not metric, in a strict sense...
class metric(object):
  def __init__(self):
    pass

  def __call__(self, clf, gen_pos, gen_neg, gen_pos_val=None, gen_neg_val=None):
    raise NotImplementedError()