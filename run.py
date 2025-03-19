import argparse
import lightning.pytorch as pl
from lightning.pytorch import loggers as plg
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.vanilla import Vanilla
from config.cfg import get_cfg_defaults
from src.utils import results_dir
pl.seed_everything(42)

args = argparse.ArgumentParser()
args.add_argument('--config', type=str, default='default.yaml')
args.add_argument('--exp_name', type=str, default='dev_1')
args.add_argument('--debug', action='store_true')
args = vars(args.parse_args())

arglist = []
for key, value in list(args.items()): 
    if value is not None: 
        arglist.append(key), arglist.append(value) 

cfg = get_cfg_defaults()
cfg.merge_from_file(f'{cfg.system.path}/config/{args["config"]}')
cfg.merge_from_list(arglist)
cfg.system.results_path = results_dir(cfg)


if not cfg.debug:
    wandb_logger = plg.WandbLogger(entity="UAVM", project="CVGL", save_dir=f'{cfg.system.results_path}/', log_model=False,
                                   name=cfg.exp_name)
    wandb_logger.log_hyperparams(cfg)
else:
    cfg.system.batch_size = 8
    cfg.model.epochs, cfg.system.workers = 25, 1
    wandb_logger = None

checkpoint_callback = ModelCheckpoint(monitor="val_epoch_loss", mode="min", dirpath=f'{cfg.system.results_path}/ckpts/', save_top_k=1)

model = Vanilla(cfg)
trainer = pl.Trainer(max_epochs=cfg.model.epochs, devices=cfg.system.workers, 
                     logger=wandb_logger if not cfg.debug else None,
                     callbacks=[checkpoint_callback],
                     check_val_every_n_epoch=4,
                     overfit_batches=4 if cfg.debug else 0,
                     num_sanity_val_steps=0,
                     default_root_dir=cfg.system.results_path,
                     )
trainer.fit(model)






# TODO: fix
# model = Vanilla.load_from_checkpoint(checkpoint_callback.best_model_path)
print(f'\n{checkpoint_callback.best_model_path}\n')
trainer = pl.Trainer(devices=1, logger=wandb_logger if not cfg.debug else None, default_root_dir=cfg.system.results_path)
trainer.test(model, ckpt_path=checkpoint_callback.best_model_path)

