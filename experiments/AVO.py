from tqdm import tqdm

import argparse
import os
import pickle

from advopt.tasks import PythiaTracker
from advopt.classifier.nn import rolling

from advopt.classifier import Network, Dense
from advopt.classifier import lincapacity, logcapacity
from advopt.classifier import l2_reg, r1_reg
from advopt.opt import variational_optimization

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--device', '-d', type=str, required=True)
  parser.add_argument('--budget', '-b', type=int, required=True)
  parser.add_argument('--seed', '-s', type=int, required=True)
  parser.add_argument('--njobs', '-n', type=int, default=8)
  parser.add_argument('--logs', type=str, default=None)
  parser.add_argument('--root', '-r', type=str, default=os.environ.get('DATA_ROOT', './'))

  arguments = parser.parse_args()
  device = arguments.device
  budget = arguments.budget
  seed = arguments.seed
  njobs = arguments.njobs
  logs = arguments.logs
  root = arguments.root

  task = PythiaTracker(seed=seed, n_jobs=njobs, logs=logs)

  const_reg = r1_reg(coef=1.0)

  net_drop = Network(
    Dense(task.ndim(), dropout=True),
    device=device, min_stagnation=1024,
    regularization=None, capacity=lincapacity(3),
    const_regularization=const_reg,
    stop_diff=1e-1
  )
  net_reg = Network(
    Dense(task.ndim(), dropout=False),
    device=device, min_stagnation=1024,
    regularization=l2_reg(1e-2), capacity=logcapacity(10),
    const_regularization=const_reg,
    stop_diff=1e-1
  )
  net_control = Network(
    Dense(task.ndim(), dropout=False),
    device=device, min_stagnation=1024,
    stop_diff=1e-1,
    regularization=None,
    const_regularization=const_reg
  )

  results_control = variational_optimization(
    metric=rolling(size_step=128, target_diff=5e-2, progress=tqdm),
    mean0=[0.75, 0.75, 0.75],
    sigma0=0.2,
    sigma_min=0.1,
    device=device
  )(task, budget, net_control, seed=111)

  results_drop = variational_optimization(
    metric=rolling(size_step=128, target_diff=5e-2, progress=tqdm),
    mean0=[0.75, 0.75, 0.75],
    sigma0=0.2,
    sigma_min=0.1,
    device=device
  )(task, budget, net_drop, seed=111)

  results_reg = variational_optimization(
    metric=rolling(size_step=128, target_diff=5e-2, progress=tqdm),
    mean0=[0.75, 0.75, 0.75],
    sigma0=0.2,
    sigma_min=0.1,
    device=device
  )(task, budget, net_reg, seed=111)

  os.makedirs(os.path.join(root, 'AVO'), exist_ok=True)

  results = {
    'dropout' : results_drop,
    'l2' : results_reg,
    'control' : results_control
  }

  with open(os.path.join(root, 'AVO', 'AVO-dense-%d-%d.pickled' % (seed, budget)), 'wb') as f:
    pickle.dump(results, f)

if __name__ == '__main__':
  main()




