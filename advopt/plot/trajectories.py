import numpy as np
import matplotlib.pyplot as plt

__all__ = [
  'plot_convergence',
  'distance_to'
]


def cum_argmin(dist, xs, fs):
  fs = np.asarray(fs)

  result = np.zeros_like(fs)
  if fs.shape[0] == 0:
    return result

  current_best = fs[0]
  result[0] = dist(xs[0])

  for i in range(1, fs.shape[0]):
    if fs[i] < current_best:
      result[i] = dist(xs[i])
      current_best = fs[i]
    else:
      result[i] = result[i - 1]

  return result

def cum_min(dist, xs):
  return cum_argmin(dist, xs, [dist(x) for x in xs])

def distance_to(x0):
  x0 = np.array(x0)

  def f(points):
    return np.sqrt(np.sum((points - x0) ** 2))

  return f

def interpolate(cs, fs, xs):
  from scipy.interpolate import interp1d
  return interp1d(cs, fs, kind='previous', fill_value='extrapolate')(xs)


def plot_convergence(
        results, dist, individual=False, budget=None, points=1024, qs=(0.1,),
        properties=None, mode='argmin', empty='ignore', extrapolate=True):
  if properties is None:
    properties = dict()

  for i, name in enumerate(results):
    if name not in properties:
      properties[name] = dict()

    properties[name]['color'] = properties[name].get('color', plt.cm.tab10(i))
    properties[name]['label'] = properties[name].get('label', name)

  trajectories = dict()

  for name in results:
    trajectories[name] = list()

    for res in results[name]:
      if len(res.points) < 2:
        if empty == 'ignore':
          continue
        else:
          raise ValueError('%s method has less than 2 points' % (name, ))

      cs = np.cumsum(res.costs)
      if mode == 'argmin':
        ds = cum_argmin(dist, res.points, res.values)
      else:
        ds = cum_min(dist, res.points)

      trajectories[name].append((cs, ds))

  if individual:
    for name in trajectories:
      for cs, ds in trajectories[name]:
        plt.plot(cs, ds, color=properties[name]['color'], alpha=0.1)

  if budget is None:
    max_cost = np.max([
      np.sum(res.costs)
      for name in results for res in results[name]
    ])
  else:
    max_cost = budget

  if extrapolate:
    xs = {
      name : np.linspace(0, max_cost, num=points)
      for name in results
    }
  else:
    xs = dict()
    for name in results:
      max_c = np.median([np.sum(r.costs) for r in results[name]])
      xs[name] = np.linspace(0, max_c, num=points)

  inter_trajectories = dict()
  for name in results:
    inter_trajectories[name] = np.ndarray(shape=(len(results[name]), points), dtype='float64')

    for i in range(len(trajectories[name])):
      cs, ds = trajectories[name][i]
      inter_trajectories[name][i, :] = interpolate(cs, ds, xs[name])

  medians = dict()
  for name in inter_trajectories:
    medians[name] = np.median(inter_trajectories[name], axis=0)

  if qs is not None:
    quantiles = dict()
    qs_ = np.sort(qs)
    qs_ = tuple(qs_) + tuple(1 - q for q in qs_[::-1])

    for name in inter_trajectories:
      quantiles[name] = np.quantile(inter_trajectories[name], axis=0, q=qs_)
  else:
    quantiles = None

  for i, name in enumerate(medians):
    plt.plot(xs[name], medians[name], **properties[name])

    if qs is not None:
      n = len(qs)
      for j, q in enumerate(qs):
        plt.fill_between(
          xs[name], quantiles[name][j, :], quantiles[name][2 * n - j - 1, :],
          alpha=q, color=properties[name]['color']
        )