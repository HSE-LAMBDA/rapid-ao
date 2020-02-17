__all__ = [
  'search'
]

class cached(object):
  def __init__(self, f):
    self._cache = dict()
    self.f = f

  def __call__(self, x):
    x = int(x)

    if x not in self._cache:
      self._cache[x] = self.f(x)

    return self._cache[x]

def search(f, xtol=32, x0=None, method='bisect', limit=None, verbose=False):
  from scipy.optimize import root_scalar

  f = cached(f)

  if x0 is None:
    x0 = xtol

  if f(x0) > 0:
    step = x0
    a, b = x0, 2 * x0

    while f(b) >= 0 and b <= limit:
      step *= 2
      a = b
      b = a + step
  else:
    a, b = 0, x0

  if b >= limit:
    return None

  if verbose:
    print('Bracket: (%d, %d)' % (a, b))
    print('Values: (%.3lf, %.3lf)' % (f(a), f(b)))

  sol = root_scalar(f, bracket=(a, b), method=method, xtol=xtol, )

  if not sol.converged:
    raise Exception('Root search did not converge.\n%s' % (sol.flag, ))

  return int(sol.root)







