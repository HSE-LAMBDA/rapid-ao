import numpy as np
from advopt.target.meta import cached_generator

def test_cached():
  n = 128
  g1 = cached_generator(lambda size: (
    np.random.normal(size=(size, 25)),
    np.random.binomial(1, 0.5, size=(size,), ),
  ))
  x1, y1 = g1(n)
  x2, y2 = g1(n)
  x3, y3 = g1(2 * n)

  assert x1.shape == (n, 25)
  assert y1.shape == (n, )

  assert x2.shape == (n, 25)
  assert y2.shape == (n,)

  assert x3.shape == (2 * n, 25)
  assert y3.shape == (2 * n,)

  assert np.allclose(x1, x2)
  assert np.allclose(y1, y2)
  assert np.allclose(x1, x3[:n])
  assert np.allclose(y1, y3[:n])

  g2 = cached_generator(lambda size:np.random.exponential(size=(size, 17)))
  x1, = g2(n)
  x2, = g2(n)
  x3, = g2(2 * n)

  assert x1.shape == (n, 17)
  assert x2.shape == (n, 17)
  assert x3.shape == (2 * n, 17)

  assert np.allclose(x1, x2)
  assert np.allclose(x1, x3[:n])

