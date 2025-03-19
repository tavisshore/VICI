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
_C.system.path = str(Path(__file__).parent.parent.resolve())
_C.system.results_path = _C.system.path + '/results'
_C.system.scheduler = 'step' # 'step' or 'plateau'

_C.data = CN()
_C.data.root = '/home/shitbox/datasets/University-Release/'
_C.data.samearea = False
_C.data.augment = True
_C.data.sample_equal = True

_C.model = CN()
_C.model.selection = 'feat'
_C.model.size = 'tiny'
_C.model.epochs = 100


def get_cfg_defaults():
  return _C.clone()