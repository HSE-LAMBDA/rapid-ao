import os
import multiprocessing as mp

import numpy as np

__all__ = [
  'experiment',
  'get_experiment_root',
  'load_results'
]

def superseed(seed=None, used_seeds=None):
  rng = np.random.RandomState(seed)

  if used_seeds is None:
    used = set()
  else:
    used = set(used_seeds)

  while True:
    s = rng.randint(0, 2 ** 31 - 1)

    while s in used:
      s = rng.randint(0, 2 ** 31 - 1)

    used.add(s)
    yield s

def worker(task, optimizer, device, command_queue, result_queue, logs='logs/', verbose=True):
  import sys
  from ..meta import apply_with_kwargs

  try:

    if logs is not None:
      sys.stdout = open(
        os.path.join(logs, '%s-%s-%d.out' % (task.name(), optimizer.name(), os.getpid())),
        mode='w'
      )
      sys.stderr = open(
        os.path.join(logs, '%s-%s-%d.err' % (task.name(), optimizer.name(), os.getpid())),
        mode='w'
      )
    else:
      sys.stdout = open(os.devnull, mode='w')
      sys.stderr = open(os.devnull, mode='w')
  except:
    import traceback
    traceback.print_exc()

  if verbose:
    log = print
  else:
    log = lambda x: x

  while True:
    try:
      command = command_queue.get()

      log()

      if command is None:
        log('Received stop command')
        return

      model_name, budget, seed = command

      log('  model: %s;\n  budget: %d;\n  seed: %d' % (model_name, budget, seed))

      model = apply_with_kwargs(
        task.models()[model_name],
        device=device,
        seed=seed
      )

      result = optimizer(task=task, model=model, budget=budget, seed=seed)
      result_queue.put((model_name, result))

    except Exception as e:
      import traceback
      stack = traceback.format_exc()

      result_queue.put((None, (e, stack)))

    sys.stdout.flush()
    sys.stderr.flush()

def load_results(root, name):
  import pickle
  import re

  r = re.compile(r'^%s-\d+[.]pickled$' % (name, ))

  try:
    results = list()
    files = [
      path
      for path in os.listdir(root)
      if r.fullmatch(path) is not None
    ]

    for path in files:
      with open(os.path.join(root, path), 'rb') as f:
        results.append(
          pickle.load(f)
        )

    return results
  except FileNotFoundError:
    return list()

def save_result(root, result, name):
  import pickle
  path = os.path.join(root, '%s-%d.pickled' % (name, result.seed))

  with open(path, 'wb') as f:
    pickle.dump(result, f)

def get_experiment_root(root, task, optimizer, budget):
  return os.path.join(
    root,
    '%s-%s-%s' % (task.name(), optimizer.name(), budget)
  )

def experiment(
  root, task, optimizer, models,
  budget, repeat,
  devices=None,
  progress=lambda x: x,
  seed=None,
  logs=None
):
  import os

  if isinstance(optimizer, str):
    optimizer = task.optimizers()[optimizer]

  root = get_experiment_root(root, task, optimizer, budget)
  os.makedirs(root, exist_ok=True)

  if logs is not None:
    os.makedirs(logs, exist_ok=True)

  results = {
    name : load_results(root, name)
    for name in models
  }

  n_tasks = sum([
    max(0, repeat - len(results[name]))
    for name in models
  ])

  if n_tasks == 0:
    return results

  used_seeds = [
    opt_result.seed
    for name in results
    for opt_result in results[name]
  ]
  seed_stream = superseed(seed, used_seeds=used_seeds)

  if devices is None:
    devices = ['cpu']

  n_procs = min(len(devices), n_tasks)

  context = mp.get_context('spawn')

  command_queue = context.Queue()
  result_queue = context.Queue()

  procs = [
    context.Process(
      target=worker,
      kwargs=dict(
        task=task,
        optimizer=optimizer,
        device=device,
        command_queue=command_queue,
        result_queue=result_queue,
        logs=logs
      )
    )

    for _, device in zip(range(n_procs), devices)
  ]

  for proc in procs:
    proc.start()

  commands = list()

  for name in models:
    left = max(0, repeat - len(results[name]))

    for _ in range(left):
      commands.append(
        (name, budget, next(seed_stream))
      )

  import random
  random.shuffle(commands)

  for command in commands:
    command_queue.put(command)

  for _ in range(n_procs):
    command_queue.put(None)

  for _ in progress(range(n_tasks)):
    name, result = result_queue.get()

    if name is None:
      e, stack = result
      print(e)
      print(stack)
    else:
      results[name].append(result)
      save_result(root, result, name)

  for proc in procs:
    proc.join(timeout=5)

    if proc.exitcode is None:
      os.kill(proc.pid, 9)

      print('Process %d was brutally killed.' % (proc.pid, ))

  return results