from yacs.config import CfgNode as CN
from pathlib import Path

_C = CN()

_C.system = CN()
_C.system.gpus = -1
_C.system.workers = 4
_C.system.path = Path.cwd()

_C.data = CN()
_C.data.root = '/home/shitbox/datasets/University-Release/'
_C.data.samearea = False
_C.data.augment = True


_C.model = CN()
_C.model.selection = 'feat'
_C.model.size = 'tiny'


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()