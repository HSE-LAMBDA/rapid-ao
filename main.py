import re

gpu_re = re.compile(r'^cuda(?:[:](\d+))?(?:[/](\d+))?$')
cpu_re = re.compile(r'^cpu(?:[/](\d+))?$')

CUDA_VISIBLE_DEVICES = 'CUDA_VISIBLE_DEVICES'

def get_device(device):
  import os

  gpu_match = gpu_re.match(device)
  if gpu_match is not None:
    device_id, n_procs = gpu_match.groups()

    if n_procs is None:
      n_procs = 1
    else:
      n_procs = int(n_procs)

    if device_id is None:
      device_name = 'cuda'
    else:
      device_name = 'cuda:%s' % (device_id, )

    return [device_name for _ in range(n_procs)]

  cpu_match = cpu_re.match(device)
  if cpu_match is not None:
    n_procs, = cpu_match.groups()

    if n_procs is None:
      n_procs = 1
    else:
      n_procs = int(n_procs)

    return ['cpu' for _ in range(n_procs)]

  raise Exception('Device spec is not understood: %s' % (device, ))

def get_devices(devices):
  return [
    dev
    for device in devices.split(',')
    for dev in get_device(device)
  ]

def main(task, optimizer, models, budget, repeat, root, devices, seed, logs=None):
  from advopt import experiment
  from tqdm import tqdm

  _ = experiment(
    root=root,
    task=task,
    models=models,
    optimizer=optimizer,
    budget=budget,
    repeat=repeat,
    devices=devices,
    progress=tqdm,
    seed=seed,
    logs=logs
  )

if __name__ == '__main__':
  from advopt import master_list

  import argparse

  parser = argparse.ArgumentParser(
    description='Rapid Adversarial Optimization experiments'
  )

  parser.add_argument('task', type=str, choices=master_list.keys())
  parser.add_argument('optimizer', type=str)
  parser.add_argument('--budget', type=int, default=1024)
  parser.add_argument('--models', type=str, nargs='*')
  parser.add_argument('--root', type=str, default='./')
  parser.add_argument(
    '--devices', type=str,
    help='''
      Specifies devices available for execution.
      For execution on GPU use: "cuda[:<GPU id>][/<n threads>]".
      For CPU: "cpu[/<number of threads>]".
      Several devices might be specified by separating them with coma.
    '''
  )
  parser.add_argument('--repeat', type=int, default=1)
  parser.add_argument('--seed', type=int, default=999111)
  parser.add_argument('--logs', type=str, default=None)

  args = parser.parse_args()

  task = master_list[args.task]()
  available_models = list(task.models().keys())

  if args.models is None or len(args.models) == 0:
    models = available_models
  elif len(args.models) == 1 and args.models[0] == 'all':
    models = available_models
  else:
    for metric in args.models:
      if metric not in available_models:
        raise Exception('Unknown metric: %s, possible choices: %s' % (metric, available_models))

    models = args.models

  optimizer = args.optimizer

  assert optimizer in task.optimizers().keys(), 'Unknown optimizer %s for the task, possible choices: %s' % (
    optimizer, task.optimizers().keys()
  )

  import os
  root = os.path.abspath(args.root)

  devices = get_devices(args.devices)
  print('Devices: %s' % (devices, ))
  import torch
  for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

  main(
    task, optimizer, models,
    budget=args.budget, repeat=args.repeat,
    root=root, devices=devices, seed=args.seed,
    logs=args.logs
  )