import lightning.pytorch as pl
from lightning.pytorch import loggers as plg
from lightning.pytorch.callbacks import ModelCheckpoint
from src.models.vanilla import Vanilla

# wandb_logger = plg.WandbLogger(project="ACMChallenge", save_dir=f'/home/shitbox/tav/challenge/lightning_logs/', log_model=False)
checkpoint_callback = ModelCheckpoint(monitor="val_epoch_loss", mode="min", dirpath='/home/shitbox/tav/challenge/ckpts/', save_top_k=1,
                                      filename='vanilla-{epoch:02d}-{val_epoch_loss:.2f}')



model = Vanilla()
trainer = pl.Trainer(max_epochs=1, devices=1, 
                    #  logger=wandb_logger,
                     callbacks=[checkpoint_callback],
                     check_val_every_n_epoch=4,
                     )
trainer.fit(model)
# print(checkpoint_callback.best_model_path)
# model = Vanilla.load_from_checkpoint(checkpoint_callback.best_model_path)
trainer.test(model)
