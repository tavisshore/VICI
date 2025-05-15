import argparse
import torch
import lightning.pytorch as pl
from src.models.vanilla import Vanilla
from config.cfg import get_cfg_defaults

args = argparse.ArgumentParser()
args.add_argument('--debug', action='store_true')
args.add_argument('--config', type=str, default='DINO-L+D.yaml')
args = vars(args.parse_args())

arglist = []
for key, value in list(args.items()): 
    if value is not None: 
        arglist.append(key), arglist.append(value) 

cfg = get_cfg_defaults()
cfg.merge_from_file(f'{cfg.system.path}/config/{args["config"]}')
cfg.merge_from_list(arglist)
cfg.system.results_path = 'results'

model = Vanilla(cfg) #.load_from_checkpoint('/scratch/projects/challenge/results/DINO-L+D.ckpt', cfg=cfg)#, strict=False)
model.model.load_state_dict(torch.load('DINO-L+D.pth'), strict=True)
model = model.eval()

trainer = pl.Trainer(devices=1, default_root_dir=cfg.system.results_path)
trainer.predict(model)
