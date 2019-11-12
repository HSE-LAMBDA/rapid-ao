import numpy as np

from advopt.target import adjusted, diff_criterion

def test_adaptive():
  from sklearn.ensemble import GradientBoostingClassifier

  tolerance = 1e-1

  clf10 = GradientBoostingClassifier(n_estimators=10, max_depth=1)
  clf100 = GradientBoostingClassifier(n_estimators=100, max_depth=1)

  gen_pos = lambda size: np.random.normal(0, 1, size=(size, 2))
  gen_neg = lambda size: np.random.normal(1, 1, size=(size, 2))

  metric = adjusted(
    criterion=diff_criterion(tolerance), xtol=32,
    verbose=True,
  )

  size1, m1 = metric.jensen_shannon(clf10, gen_pos, gen_neg)
  size2, m2 = metric.jensen_shannon(clf10, gen_pos, gen_neg)

  assert np.abs(m1 - m2) < 2 * tolerance

  print(size1, m1)
  print(size2, m2)

  size1, m1 = metric.jensen_shannon(clf100, gen_pos, gen_neg)
  size2, m2 = metric.jensen_shannon(clf100, gen_pos, gen_neg)

  assert np.abs(m1 - m2) < 2 * tolerance

  gen_pos = lambda size: (
    np.random.normal(0, 1, size=(size, 2)),
    np.random.binomial(1, 0.5, size=(size, ))
  )
  gen_neg = lambda size: (
    np.random.normal(1, 1, size=(size, 2)),
    np.random.binomial(1, 0.5, size=(size,))
  )

  size, jsd = metric.jensen_shannon(clf10, gen_pos, gen_neg)
  print(size, jsd)
  size, jsd = metric.jensen_shannon(clf10, gen_pos, gen_neg)
  print(size, jsd)

# def test_adaclf():
#   from sklearn.ensemble import GradientBoostingClassifier
#
#   clf = GradientBoostingClassifier(n_estimators=20, max_depth=3)
#   adaclf = AdaptiveBoosting(
#     GradientBoostingClassifier(n_estimators=20, max_depth=3),
#     thresholds=np.linspace(np.log(2) / 2, np.log(2), num=20)
#   )
#
#   metric = adjusted(
#     jensen_shannon(clf),
#     criterion=diff_criterion(1e-2), xtol=32
#   )
#
#   adametric = adjusted(
#     jensen_shannon(adaclf),
#     criterion=diff_criterion(1e-2), xtol=32
#   )
#
#   gen_pos = lambda size: np.random.normal(0, 1, size=(size, 2))
#   gen_neg = lambda x: lambda size: np.random.normal(x, 1, size=(size, 2))
#
#
#   size, value = metric(gen_pos, gen_neg(0))
#   print('Classifier:')
#   print('%.3lf [%d]' % (value, size))
#   print()
#   size, value = adametric(gen_pos, gen_neg(0))
#   print('Adaptive Classifier [%d]:' % (adaclf.capacity, ))
#   print('%.3lf [%d]' % (value, size))
#
#   assert False
#
# def test_boosting():
#   from sklearn.ensemble import GradientBoostingClassifier
#
#   n = 20
#
#   jsd0 = np.log(2)
#
#   metrics = dict(
#     metric = adjusted(
#       jensen_shannon(GradientBoostingClassifier(n_estimators=n, max_depth=3)),
#       criterion=diff_criterion(1e-2), xtol=32
#     ),
#
#     linmetric = adjusted(
#       jensen_shannon(
#         AdaptiveBoosting(
#           GradientBoostingClassifier(n_estimators=n, max_depth=3),
#           thresholds=linspace(jsd0, 0, num=n))
#         ),
#       criterion=diff_criterion(1e-2), xtol=32
#     ),
#
#     expmetric = adjusted(
#       jensen_shannon(
#         AdaptiveBoosting(
#           GradientBoostingClassifier(n_estimators=n, max_depth=3),
#           thresholds=expspace(jsd0, 0, num=n))
#       ),
#       criterion=diff_criterion(1e-2), xtol=32
#     ),
#
#     logmetric = adjusted(
#       jensen_shannon(
#         AdaptiveBoosting(
#           GradientBoostingClassifier(n_estimators=n, max_depth=3),
#           thresholds=logspace(jsd0, 1e-3, num=n))
#       ),
#       criterion=diff_criterion(1e-2), xtol=32
#     )
#   )
#
#   gen_pos = lambda size: np.random.normal(0, 1, size=(size, 1))
#   gen_neg = lambda mean, sigma: lambda size: np.random.normal(mean, sigma, size=(size, 1))
#
#   means = np.linspace(-2, 2, num=3)
#   sigmas = np.linspace(0, 2, num=5)
#
#   sizes = dict([
#     (name, np.zeros(shape=(means.shape[0], sigmas.shape[0])))
#     for name in metrics
#   ])
#   values = dict([
#     (name, np.zeros(shape=(means.shape[0], sigmas.shape[0])))
#     for name in metrics
#   ])
#
#   for name in metrics:
#     for i, m in enumerate(means):
#       for j, s in enumerate(sigmas):
#         sizes[name][i, j], values[name][i, j] = metrics[name](gen_pos, gen_neg(m, s))
#
#   import matplotlib.pyplot as plt
#
#   plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
#   plt.suptitle(r'$\mu(\mathcal{N}(0, 1), \mathcal{N}(\mu, \sigma^2)$')
#   for k, name in enumerate(metrics.keys()):
#     plt.subplot(2, 2, k + 1)
#     plt.contourf(means, sigmas, values[name].T, levels=np.linspace(0, np.log(2), num=10))
#     plt.colorbar()
#     plt.ylabel(r'$\sigma$')
#     plt.xlabel(r'$\mu$')
#     plt.title(name)
#
#   plt.savefig('JSD.png')
#   plt.show()
#
#   max_size = np.max([ np.max(sizes[name]) for name in sizes ])
#
#   plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
#   plt.suptitle(r'$\mu(\mathcal{N}(0, 1), \mathcal{N}(\mu, \sigma^2)$')
#   for k, name in enumerate(metrics.keys()):
#     plt.subplot(2, 2, k + 1)
#     plt.contourf(means, sigmas, sizes[name].T, levels=np.linspace(0, max_size, num=10))
#     plt.colorbar()
#     plt.ylabel(r'$\sigma$')
#     plt.xlabel(r'$\mu$')
#     plt.title(name)
#
#   plt.savefig('JSD-sizes.png')
#   plt.show()
#
#   assert False
