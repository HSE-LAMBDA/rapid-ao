import numpy as np

from advopt.target.search import cached

def test_compare():
  from scipy.optimize import root_scalar
  methods = ['bisect', 'brentq', 'brenth', 'ridder', 'toms748']

  errors = dict([ (name, list()) for name in methods ])
  n_iters = dict([(name, list()) for name in methods])

  for _ in range(100):
    w = 10 ** np.random.uniform(1, 2)
    c = np.random.uniform(0.1, np.log(2))
    f0 = 1e-3

    solution = -np.log(f0 / c) / w

    f = lambda x: c * np.exp(-w * x) - f0

    x1 = 100
    while f(x1) > -f0 / 2:
      x1 *= 10

    for method in methods:
      f_c = cached(f)
      sol = root_scalar(f_c, bracket=(0, x1), method=method, maxiter=100, xtol=10)
      errors[method].append(np.abs(sol.root - solution))
      n_iters[method].append(np.sum(list(f_c.cache.keys())))


  for method in methods:
    print(
      '%s: %.3lf +- %.3lf [%.1lf +- %.1lf]' % (
        method.ljust(10),
        np.mean(errors[method]), np.std(errors[method]),
        np.mean(n_iters[method]), np.std(n_iters[method]),
      )
    )

  assert False
