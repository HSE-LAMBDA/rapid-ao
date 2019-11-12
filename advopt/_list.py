from . import tasks

__all__ = [
  'master_list'
]

master_list = {
  'roll' : lambda : tasks.SwissRoll(),
  'xor' : lambda : tasks.XOR(),
  'tunemc' : lambda : tasks.PythiaTuneMC(n_params=None, n_jobs=8),
  'tunemc1' : lambda : tasks.PythiaTuneMC(n_params=1, n_jobs=8),
  'pythia-tracker' :  lambda : tasks.PythiaTracker(n_jobs=8),
}