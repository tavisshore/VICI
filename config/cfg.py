from yacs.config import CfgNode as CN
from pathlib import Path

_C = CN()
_C.exp_name = 'exp_name'
_C.debug = False
_C.config = 'default.yaml'

_C.system = CN()
_C.system.gpus = -1
_C.system.workers = 4
_C.system.batch_size = 16
_C.system.path = str(Path.cwd())
_C.system.results_path = str(Path.cwd()) + '/results'

_C.data = CN()
_C.data.root = '/home/shitbox/datasets/University-Release/'
_C.data.samearea = False
_C.data.augment = True
_C.data.train_prop = 0.9
_C.data.sample_equal = True

_C.model = CN()
_C.model.selection = 'feat'
_C.model.size = 'tiny'
_C.model.epochs = 200


def get_cfg_defaults():
  return _C.clone()