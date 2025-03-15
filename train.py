import lightning.pytorch as pl
from torch.utils.data import DataLoader
from lightning.pytorch import loggers as plg
from src.models.vanilla import Vanilla
from src.data.uni import University1652_CVGL

wandb_logger = plg.WandbLogger(project="ACMChallenge", save_dir=f'lightning_logs/', log_model=False)

train_dataset = University1652_CVGL()
val_dataset = University1652_CVGL(stage='val')
test_dataset = University1652_CVGL(stage='test')
train = DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)
val = DataLoader(val_dataset, batch_size=16, num_workers=4, shuffle=False)
test = DataLoader(test_dataset, batch_size=1, num_workers=4, shuffle=False)

model = Vanilla()
trainer = pl.Trainer(max_epochs=100, devices=4, 
                     logger=wandb_logger,
                     check_val_every_n_epoch=1,
                     )
trainer.fit(model, train, val)
trainer.test(model, test)
