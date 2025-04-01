import argparse
import lightning.pytorch as pl
from lightning.pytorch import loggers as plg
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.tuner.tuning import Tuner
from src.models.vanilla import Vanilla
from src.models.ssl import SSL
from config.cfg import get_cfg_defaults
from src.utils import results_dir


pl.seed_everything(42)

args = argparse.ArgumentParser()
args.add_argument('--debug', action='store_true')
args.add_argument('--config', type=str, default='default.yaml')
args = vars(args.parse_args())

arglist = []
for key, value in list(args.items()): 
    if value is not None: 
        arglist.append(key), arglist.append(value) 

cfg = get_cfg_defaults()
cfg.merge_from_file(f'{cfg.system.path}/config/{args["config"]}')
cfg.merge_from_list(arglist)

# Why does this create multiple dirs - number of workers? devices??
cfg.system.results_path = results_dir(cfg)


if not cfg.debug:
    # This is a dumb one. Don't know why DDP on AMD gonna freeze wandb logger
    # So choose to disable it on AMD cluster while on DDP
    if cfg.system.amd_cluster:
        wandb_logger = None
    else:
        wandb_logger = plg.WandbLogger(entity="UAVM", project="CVGL", save_dir=f'{cfg.system.results_path}/', log_model=False, name=cfg.exp_name)
        wandb_logger.log_hyperparams(cfg)

else:
    cfg.system.batch_size = 32
    cfg.model.epochs = 5
    cfg.system.gpus = 1
    wandb_logger = None

checkpoint_callback = ModelCheckpoint(monitor="val_mean", mode="max", dirpath=f'{cfg.system.results_path}/ckpts/', save_top_k=1, filename='{epoch}-{val_mean:.2f}')

# model = Vanilla(cfg)
model = SSL(cfg)

trainer = pl.Trainer(max_epochs=cfg.model.epochs, devices=cfg.system.gpus, 
                     logger=wandb_logger if not cfg.debug else None,
                     callbacks=[checkpoint_callback],
                     check_val_every_n_epoch=2,
                     overfit_batches=4 if cfg.debug else 0,
                     num_sanity_val_steps=0,
                     strategy='auto',
                     default_root_dir=cfg.system.results_path,
                    #  gradient_clip_val=0.5, 
                    #  gradient_clip_algorithm="value",
                     )

if cfg.system.gpus == 1 and not cfg.debug:
    tuner = Tuner(trainer)
    if cfg.system.tune.lr: cfg.model.lr = tuner.lr_find(model)
    if cfg.system.tune.batch_size: tuner.scale_batch_size(model, mode='power', init_val=cfg.system.batch_size)
        
trainer.fit(model)

###
# TODO: re-write here to use rank_0 decorate to have more elegant code
if trainer.local_rank == 0:
    trainer = pl.Trainer(devices=1, default_root_dir=cfg.system.results_path, callbacks=[checkpoint_callback])
    trainer.validate(model, ckpt_path=checkpoint_callback.best_model_path)
    trainer.test(model, ckpt_path=checkpoint_callback.best_model_path)
else: # Nothing to do in other rank, just put as a barrier here.
    pass