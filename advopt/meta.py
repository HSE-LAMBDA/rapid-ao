import inspect

__all__ = [
  'classfunc',
  'apply_with_kwargs'
]

def get_kwargs(func):
  params = inspect.signature(func).parameters
  return [p.name for p in params.values()]

def apply_with_kwargs(f, *args, **kwargs):
  accepted_kwargs = get_kwargs(f)
  passed_kwargs = dict()

  for k, v in kwargs.items():
    if k in accepted_kwargs:
      passed_kwargs[k] = v

  return f(*args, **passed_kwargs)

def classfunc(f):
  def __init__(self, *args, **kwargs):
    self.args = args
    self.kwargs = kwargs

  def __call__(self, *args, **kwargs):
    return f(*self.args, **self.kwargs)(*args, **kwargs)

  clazz = type(
    f.__name__,
    (object, ),
    dict(__init__=__init__, __call__=__call__)
  )

  clazz.__init__.__signature__ = inspect.signature(f)

  globals()[f.__name__] = clazz

  return clazz