import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import timm 
from pytorch_metric_learning import losses
from pytorch_metric_learning.miners import BatchEasyHardMiner
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from torch.nn import functional as F
import zipfile
from dotmap import DotMap
from copy import deepcopy
from src.data.uni import University1652_CVGL
from src.utils import recall_accuracy, get_backbone
from src.data.database import ImageDatabase

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias = True, use_bn = False, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.use_bn = use_bn
        
        self.linear = nn.Linear(self.in_features, self.out_features, bias = self.use_bias and not self.use_bn)
        if self.use_bn:
             self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self,x) -> torch.Tensor:
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, head_type = 'nonlinear', **kwargs):
        super(ProjectionHead,self).__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.head_type = head_type

        if self.head_type == 'linear':
            self.layers = LinearLayer(self.in_features, self.out_features, False, True)
        elif self.head_type == 'nonlinear':
            self.layers = nn.Sequential(LinearLayer(self.in_features, self.hidden_features, True, True),
                                        nn.ReLU(),
                                        LinearLayer(self.hidden_features, self.out_features, False, True))
    
    def forward(self,x) -> torch.Tensor:
        x = self.layers(x)
        return x


class FeatureExtractor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.street_conv = get_backbone(cfg)
        if not cfg.model.shared_extractor:
            self.sat_conv = deepcopy(self.street_conv)

        self.data_config = timm.data.resolve_model_data_config(self.street_conv)

        if cfg.model.head.use:
            self.street_head = ProjectionHead(cfg.mode.head.params.inter_dims, cfg.mode.head.params.hidden_dims, cfg.mode.head.params.output_dims)
            if not cfg.model.shared_extractor:
                self.sat_head = deepcopy(self.street_head)

    def embed_street(self, pov_tile: torch.Tensor) -> torch.Tensor: 
        x = self.street_conv(pov_tile)
        if self.cfg.model.head.use:
            x = self.street_head(x)
        return x
    
    def embed_sat(self, map_tile: torch.Tensor) -> torch.Tensor: 
        x = self.sat_conv(map_tile)
        if self.cfg.model.head.use:
            x = self.sat_head(x)
        return x
    

class Vanilla(pl.LightningModule):
    def __init__(self, cfg):
        super(Vanilla, self).__init__()
        self.cfg = cfg
        self.lr = cfg.model.lr
        self.batch_size = cfg.system.batch_size

        self.lmdb_dataset = ImageDatabase(path=cfg.data.root)
        
        self.model = FeatureExtractor(cfg)
        self.model.to(device)

        if cfg.model.miner:
            assert cfg.model.miner in ['easy', 'semihard', 'hard', 'all'], f"Miner {cfg.model.miner} not supported"
            self.miner_func = BatchEasyHardMiner(pos_strategy=cfg.model.miner, neg_strategy=cfg.model.miner)
        else:
            self.miner_func = self.exhaustive_triplets

        self.loss_func = losses.NTXentLoss(temperature=0.1)
        self.mse = nn.MSELoss()
        
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.train_query, self.train_ref = [], []
        self.val_query, self.val_ref = [], []
        self.train_labels, self.val_labels = [], []
        self.test_outputs = DotMap(streetview=DotMap(), satellite=DotMap())

    def train_dataloader(self):
        train_dataset = University1652_CVGL(self.cfg, stage='train', data_config=self.model.data_config, lmdb=self.lmdb_dataset)
        return DataLoader(train_dataset, batch_size=self.cfg.system.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        val_dataset = University1652_CVGL(self.cfg, stage='val', data_config=self.model.data_config, lmdb=self.lmdb_dataset)
        return DataLoader(val_dataset, batch_size=self.cfg.system.batch_size, num_workers=4, shuffle=False)
    
    def test_dataloader(self):
        self.test_dataset = University1652_CVGL(self.cfg, stage='test', data_config=self.model.data_config, lmdb=self.lmdb_dataset)
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4, shuffle=False)
    
    def forward(self, street: torch.Tensor = None, sat: torch.Tensor = None, image: torch.Tensor = None, 
                branch: str = 'streetview', stage: str = 'train'):
        if stage == 'test':
            if branch == 'streetview' or self.cfg.model.shared_extractor:
                x = self.model.embed_street(image)
            else:
                x = self.model.embed_sat(image)
            return F.normalize(x, p=2, dim=1)
    
        street_out = self.model.embed_street(street)
        if self.cfg.model.shared_extractor:
            sat_out = self.model.embed_street(sat)
        else:
            sat_out = self.model.embed_sat(sat) 
        
        street_out = F.normalize(street_out, p=2, dim=1)
        sat_out = F.normalize(sat_out, p=2, dim=1)
        return street_out, sat_out

    def exhaustive_triplets(self, street_out=None, labs=None, sat_out=None, ref_labs=None):
        # Changing to (pos_pairs, neg_pairs) for NTXentLoss
        batch_size = street_out.shape[0]

        oppose = batch_size - 1
        anchors = torch.arange(batch_size).repeat_interleave(oppose)
        positives = torch.arange(batch_size, 2 * batch_size).repeat_interleave(oppose)
        negatives = torch.stack([
            torch.cat((torch.arange(i), torch.arange(i + 1, batch_size)))
            for i in range(batch_size)
        ]).reshape(batch_size, -1)[:, :oppose].flatten()

        label_tensor = torch.tensor([hash(label) for label in labs])  
        mask = label_tensor[anchors] != label_tensor[negatives]
        anchors, positives, negatives = anchors[mask], positives[mask], negatives[mask]
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        negatives += batch_size

        return anchors, positives, negatives

    def training_step(self, batch, batch_idx):
        street, sat = batch['streetview'], batch['satellite']
        id_labels = batch['label']
        sat = sat.to(device)
        street = street.to(device)

        street_out, sat_out = self(street, sat)
        
        if self.cfg.model.miner:
            labs = torch.arange(start=0, end=street_out.shape[0])
            mined_indices = self.miner_func(street_out, labs, sat_out, labs)
            loss = self.loss_func(embeddings=street_out, labels=mined_indices[2], ref_emb=sat_out, ref_labels=mined_indices[3])
        else:
            embs = torch.cat((street_out.float(), sat_out.float()), dim=0)
            mined_indices = self.miner_func(street_out, id_labels)
            loss = self.loss_func(embs, indices_tuple=mined_indices)

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=street.shape[0])
        self.train_loss.append(loss) 
        self.train_query.append([x.cpu().detach().numpy() for x in street_out])
        self.train_ref.append([x.cpu().detach().numpy() for x in sat_out])
        self.train_labels.append(id_labels)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x for x in self.train_loss]).mean()
        self.log('train_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        train_labels = [item for sublist in self.train_labels for item in sublist]
        query = np.concatenate(self.train_query, axis=0)
        ref = np.concatenate(self.train_ref, axis=0)

        metrics = recall_accuracy(query, ref, train_labels)
        for i in [1, 5, 10]: self.log(f'train_{i}', metrics[i], on_epoch=True, prog_bar=False, logger=True, sync_dist=True) 
        self.train_loss, self.train_query, self.train_ref = [], [], []
        self.train_labels = []
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        street, sat = batch['streetview'], batch['satellite']
        sat = sat.to(device)
        street = street.to(device)

        street_out, sat_out = self(street, sat)
        loss = self.mse(street_out, sat_out)

        self.log('val_loss', loss, on_step=True, prog_bar=False, logger=True, sync_dist=True, batch_size=street.shape[0])
        self.val_loss.append(loss)
        self.val_query.append([x.cpu().detach().numpy() for x in street_out])
        self.val_ref.append([x.cpu().detach().numpy() for x in sat_out])
        self.val_labels.append(batch['label'])
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.val_loss]).mean()
        self.log('val_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        val_labels = [item for sublist in self.val_labels for item in sublist]
        query = np.concatenate(self.val_query, axis=0)
        ref = np.concatenate(self.val_ref, axis=0)
        metrics = recall_accuracy(query, ref, val_labels)
        for i in [1, 5, 10]: self.log(f'val_{i}', metrics[i], on_epoch=True, prog_bar=False, logger=True, sync_dist=True) 

        mean_val_1_10 = torch.stack([self.trainer.callback_metrics[f'val_{i}'] for i in [1, 10]]).mean()
        self.log('val_mean', mean_val_1_10, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.val_loss, self.val_query, self.val_ref = [], [], []
        self.val_labels = []

        current_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log('current_lr', current_lr, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        batch_keys = batch.keys()
        branch = 'streetview' if 'streetview' in batch_keys else 'satellite'
        image = batch[branch]
        image = image.to(device)
        x_out = self.forward(image=image, branch=branch, stage='test')
        x_out = x_out.cpu().detach().numpy()
        self.test_outputs[branch][batch['name'][0]] = x_out

    def on_test_epoch_end(self):
        # Get top-10 retrievals for each streetview image and save names to file
        streetview_keys = self.test_dataset.test_keys
        satellite_keys = list(self.test_outputs['satellite'].keys())
        streetview_embeddings = [self.test_outputs['streetview'][x] for x in streetview_keys]
        satellite_embeddings = [self.test_outputs['satellite'][x] for x in satellite_keys]
        streetview = np.concatenate(streetview_embeddings)
        satellite = np.concatenate(satellite_embeddings)

        # Calculate cosine similarity between streetview and satellite embeddings
        similarity = np.dot(streetview, satellite.T)
        similarity = np.argsort(similarity, axis=1)

        # Save top-10 retrievals to file
        answer_file = f'{self.cfg.system.results_path}/answer.txt'
        with open(answer_file, 'w') as f:
            for sim in similarity:
                for s in sim[:10]:
                    f.write(f"{satellite_keys[s]}\t")
                f.write("\n")
        f.close()

        # Zip answer file
        loczip = f'{self.cfg.system.results_path}/answer.zip'
        with zipfile.ZipFile(loczip, "w", compression=zipfile.ZIP_STORED) as myzip:
            myzip.write(answer_file, arcname="answer.txt")
        myzip.close()

        # Save config & model
        self.trainer.save_checkpoint(f"{self.cfg.system.results_path}/final_model.ckpt")
        with open(f"{self.cfg.system.results_path}/config.yaml", "w") as f:
            f.write(self.cfg.dump())
        
    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=self.lr) 
        if self.cfg.system.scheduler == 'plateau':
            sch = ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5)
            return [opt], [{"scheduler": sch, "interval": "epoch", 'frequency': 5, "monitor": "val_loss_epoch"}]
        elif self.cfg.system.scheduler == 'step':
            sch = StepLR(optimizer=opt, step_size=40, gamma=0.5)
            return [opt], [{"scheduler": sch, "interval": "epoch"}]
        elif self.cfg.system.scheduler == 'cos':
            sch = CosineAnnealingLR(optimizer=opt, T_max=self.cfg.model.epochs)
            return [opt], [{"scheduler": sch, "interval": "epoch"}]
        else:
            return opt

