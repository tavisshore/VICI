import lightning.pytorch as pl
from lightning.pytorch import loggers as plg
from lightning.pytorch.callbacks import ModelCheckpoint

from src.models.vanilla import Vanilla
from config.cfg import get_cfg_defaults
pl.seed_everything(42)


cfg = get_cfg_defaults()
cfg.merge_from_file('config/default.yaml')
cfg.freeze()

debug = False
epochs, devs = 1, 1
wandb_logger = None
if not debug:
    epochs = 100
    devs = 4
    wandb_logger = plg.WandbLogger(entity="UAVM", project="CVGL", save_dir=f'{cfg.system.path}/lightning_logs/', log_model=False,
                                   name='crossarea'
                                   )
checkpoint_callback = ModelCheckpoint(monitor="val_epoch_loss", mode="min", dirpath=f'{cfg.system.path}/ckpts/', save_top_k=1,
                                      filename='vanilla-{epoch:02d}-{val_epoch_loss:.6f}')

model = Vanilla(cfg)
trainer = pl.Trainer(max_epochs=epochs, devices=devs, 
                     logger=wandb_logger if not debug else None,
                     callbacks=[checkpoint_callback],
                     check_val_every_n_epoch=4,
                     log_every_n_steps=1,
                     overfit_batches=10,
                     )
trainer.fit(model)
# TODO: fix
# model = Vanilla.load_from_checkpoint(checkpoint_callback.best_model_path)
trainer.test(model)
