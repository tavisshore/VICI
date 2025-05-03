from yacs.config import CfgNode as CN
from pathlib import Path
import getpass

# Alter dataset root path

_C = CN()
_C.exp_name = 'exp_name'
_C.debug = False
_C.config = 'default.yaml'
_C.data_config = CN()

_C.system = CN()
_C.system.gpus = 1
_C.system.workers = 4
_C.system.path = str(Path(__file__).parent.parent.resolve())
_C.system.tune = CN()
_C.system.tune.lr = False
_C.system.tune.batch_size = True
_C.system.results_path = _C.system.path + '/results/'
_C.system.scheduler = 'cos' # 'step' or 'plateau' or 'cos'
_C.system.batch_size = 8

# This is a dumb paramerter. 
# But I need to disable wandb logger in this cluser to avoid freezing
_C.system.amd_cluster = False


_C.data = CN()
_C.data.root = f'/scratch/datasets/University/'
_C.data.query_file = "src/data/query_street_name.txt"
_C.data.type = 'folder' # 'lmdb', 'folder', #'ffcv
_C.data.sample_equal = True



_C.model = CN()
_C.model.epochs = 10
_C.model.lr = 1e-4

_C.model.backbone = 'convnext' # 'convnext', 'dinov2
_C.model.size = 'tiny'
_C.model.image_size = 384
_C.model.shared_extractor = False
_C.model.miner = False


_C.model.head = CN()
_C.model.head.use = False
_C.model.head.params = CN()
_C.model.head.params.inter_dims = 1024 # MAKE THIS MATCH YOUR MODEL OUTPUT
_C.model.head.params.hidden_dims = 1024
_C.model.head.params.output_dims = 1024


def get_cfg_defaults():
  return _C.clone()



