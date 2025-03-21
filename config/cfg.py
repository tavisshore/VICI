from yacs.config import CfgNode as CN
from pathlib import Path

# Alter dataset root path

_C = CN()
_C.exp_name = 'exp_name'
_C.debug = False
_C.config = 'default.yaml'
_C.data_config = CN()

_C.system = CN()
_C.system.gpus = -1
_C.system.workers = 4
_C.system.path = str(Path(__file__).parent.parent.resolve())
_C.system.results_path = _C.system.path + '/results/'
_C.system.scheduler = 'step' # 'step' or 'plateau' or 'cos'

_C.data = CN()
_C.data.root = '/home/shitbox/datasets/University-Release/'
_C.data.samearea = False
_C.data.augment = True
_C.data.sample_equal = True


_C.model = CN()
_C.model.epochs = 200

_C.model.backbone = 'convnext' # 'convnext', 'dinov2
_C.model.size = 'tiny'

_C.model.image_size = 384
_C.system.batch_size = 32
_C.model.shared_extractor = False
_C.model.miner = 'hard'


_C.model.head = CN()
_C.model.head.use = False
_C.model.head.params = CN()
_C.model.head.params.inter_dims = 1024 # MAKE THIS MATCH YOUR MODEL OUTPUT
_C.model.head.params.hidden_dims = 1024
_C.model.head.params.output_dims = 1024


def get_cfg_defaults():
  return _C.clone()



