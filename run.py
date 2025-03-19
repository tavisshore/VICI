import argparse
import lightning.pytorch as pl
from lightning.pytorch import loggers as plg
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.vanilla import Vanilla
from config.cfg import get_cfg_defaults
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

if not cfg.debug:
    wandb_logger = plg.WandbLogger(entity="UAVM", project="CVGL", save_dir=f'{cfg.system.path}/lightning_logs/', log_model=False,
                                   name=cfg.exp_name)
    wandb_logger.log_hyperparams(cfg)
    checkpoint_callback = ModelCheckpoint(monitor="val_epoch_loss", mode="min", dirpath=f'{cfg.system.path}/ckpts/', save_top_k=1,
                                        filename='vanilla-{epoch:02d}-{val_epoch_loss:.6f}')
else:
    cfg.system.batch_size = 5
    cfg.model.epochs, cfg.system.workers = 1, 1
    wandb_logger = None

model = Vanilla(cfg)
trainer = pl.Trainer(max_epochs=cfg.model.epochs, devices=cfg.system.workers, 
                     logger=wandb_logger if not cfg.debug else None,
                     callbacks=[checkpoint_callback] if not cfg.debug else None,
                     check_val_every_n_epoch=4,
                     overfit_batches=4 if cfg.debug else 0,
                     num_sanity_val_steps=0
                     )
trainer.fit(model)
# TODO: fix
# model = Vanilla.load_from_checkpoint(checkpoint_callback.best_model_path)
results_folder = trainer.test(model)

# save config 
# with open(f"{results_folder}/config.yaml", "w") as f:
#   f.write(cfg.dump())
