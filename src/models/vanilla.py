from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import timm 
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights
from pytorch_metric_learning import losses, miners
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

from torch.nn import functional as F
import zipfile
from dotmap import DotMap

from src.data.uni import University1652_CVGL
from src.utils import recall_accuracy

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

    def forward(self,x):
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
    
    def forward(self,x):
        x = self.layers(x)
        return x

class SharedConvNextExtractor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.model.size == 'tiny':
            self.shared_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.shared_conv.classifier[2] = nn.Identity()
            if cfg.model.head.use:
                assert self.cfg.mode.head.params.inter_dims == 768, f"Inter dims should be 768 for tiny model, but got {self.cfg.mode.head.params.inter_dims}"
        elif cfg.model.size == 'base':
            self.shared_conv = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            self.shared_conv.classifier[2] = nn.Identity()
            if cfg.model.head.use:
                assert self.cfg.mode.head.params.inter_dims == 1024, f"Inter dims should be 1024 for base model, but got {self.cfg.mode.head.params.inter_dims}"

        # Add projection head
        if cfg.model.head.use:
            self.proj_head = ProjectionHead(cfg.mode.head.params.inter_dims, cfg.mode.head.params.hidden_dims, cfg.mode.head.params.output_dims)
        
    def embed_street(self, pov_tile: torch.Tensor) -> torch.Tensor: 
        x = self.shared_conv(pov_tile)
        if self.cfg.model.head.use:
            x = self.proj_head(x)
        return x
    
    def embed_sat(self, map_tile: torch.Tensor) -> torch.Tensor: 
        x = self.shared_conv(map_tile)
        if self.cfg.model.head.use:
            x = self.proj_head(x)
        return x

class ConvNextExtractor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.model.backbone == 'convnext': # convnextv2_huge.fcmae_ft_in22k_in1k_512
            self.street_conv = timm.create_model(f'timm/convnextv2_{cfg.model.size}.fcmae_ft_in22k_in1k_{cfg.model.image_size}', pretrained=True, num_classes=0)
            self.sat_conv = timm.create_model(f'timm/convnextv2_{cfg.model.size}.fcmae_ft_in22k_in1k_{cfg.model.image_size}', pretrained=True, num_classes=0)
            # self.street_conv = timm.create_model(f'convnext_{cfg.model.size}.fb_in22k_ft_in1k_384', pretrained=True, num_classes=0)
            # self.sat_conv = timm.create_model(f'convnext_{cfg.model.size}.fb_in22k_ft_in1k_384', pretrained=True, num_classes=0)
            # convnextv2_base.fcmae_ft_in22k_in1k
        elif cfg.model.backbone == 'dinov2':
            self.street_conv = timm.create_model(f'timm/vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
            self.sat_conv = timm.create_model(f'timm/vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
        elif cfg.model.backbone == 'vit':
            self.street_conv = timm.create_model(f'timm/vit_base_patch16_siglip_512.v2_webli', pretrained=True, num_classes=0)
            self.sat_conv = timm.create_model(f'timm/vit_base_patch16_siglip_512.v2_webli', pretrained=True, num_classes=0)

        self.data_config = timm.data.resolve_model_data_config(self.street_conv)
        # Get dims from this config?
        # if cfg.model.head.use:
            # assert self.cfg.mode.head.params.inter_dims == 768 if cfg.model.size == 'tiny' else 1024, f"Inter dims should be 768 for tiny model, but got {self.cfg.mode.head.params.inter_dims}"

        # Add projection head
        if cfg.model.head.use:
            self.street_head = ProjectionHead(cfg.mode.head.params.inter_dims, cfg.mode.head.params.hidden_dims, cfg.mode.head.params.output_dims)
            self.sat_head = ProjectionHead(cfg.mode.head.params.inter_dims, cfg.mode.head.params.hidden_dims, cfg.mode.head.params.output_dims)
        
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

        if cfg.model.shared_extractor:
            self.model = SharedConvNextExtractor(cfg)
        else:
            self.model = ConvNextExtractor(cfg)
        self.model.to(device)

        self.loss_func = losses.NTXentLoss()
        self.mse = nn.MSELoss()

        self.miner = None

        if cfg.model.miner == 'hard':
            self.miner_func = miners.BatchEasyHardMiner(
                pos_strategy='hard', 
                neg_strategy='hard'
                )
        
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.train_query, self.train_ref = [], []
        self.val_query, self.val_ref = [], []
        self.test_outputs = DotMap(streetview=DotMap(), satellite=DotMap())

    def train_dataloader(self):
        train_dataset = University1652_CVGL(self.cfg, stage='train', data_config=self.model.data_config)
        return DataLoader(train_dataset, batch_size=self.cfg.system.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        val_dataset = University1652_CVGL(self.cfg, stage='val', data_config=self.model.data_config)
        return DataLoader(val_dataset, batch_size=self.cfg.system.batch_size, num_workers=4, shuffle=False)
    
    def test_dataloader(self):
        self.test_dataset = University1652_CVGL(self.cfg, stage='test', data_config=self.model.data_config)
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4, shuffle=False)
    
    def forward(self, street: torch.Tensor = None, sat: torch.Tensor = None, image: torch.Tensor = None, branch: str = 'streetview', stage: str = 'train'):
        if stage == 'test':
            if branch == 'streetview':
                x = self.model.embed_street(image)
            else:
                x = self.model.embed_sat(image)
            x = F.normalize(x, p=2, dim=1)
            return x
    
        street_out = self.model.embed_street(street)
        sat_out = self.model.embed_sat(sat)
        street_out = F.normalize(street_out, p=2, dim=1)
        sat_out = F.normalize(sat_out, p=2, dim=1)
        return street_out, sat_out

    def exhaustive_triplets(self, street: torch.Tensor, sat: torch.Tensor, labels: torch.Tensor):
        embeddings = torch.cat((street.float(), sat.float()), dim=0)
        batch_size = street.shape[0]

        oppose = batch_size - 1
        anchors = torch.arange(batch_size).repeat_interleave(oppose)
        positives = torch.arange(batch_size, 2 * batch_size).repeat_interleave(oppose)
        negatives = torch.stack([
            torch.cat((torch.arange(i), torch.arange(i + 1, batch_size)))
            for i in range(batch_size)
        ]).reshape(batch_size, -1)[:, :oppose].flatten()

        label_tensor = torch.tensor([hash(label) for label in labels])  
        mask = label_tensor[anchors] != label_tensor[negatives]
        anchors, positives, negatives = anchors[mask], positives[mask], negatives[mask]
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        negatives += batch_size
        return embeddings, anchors, positives, negatives

    def training_step(self, batch, batch_idx):
        street, sat = batch['streetview'], batch['satellite']
        labels = batch['label']
        sat = sat.to(device)
        street = street.to(device)

        street_out, sat_out = self(street, sat)
        embs, anchors, positives, negatives = self.exhaustive_triplets(street_out, sat_out, labels)

        if self.miner != None:
            miner_output = self.miner_func(embs, labels)
            loss = self.loss_func(embs, indices_tuple=miner_output)
        else:
            loss = self.loss_func(embs, indices_tuple=(anchors, positives, negatives))

        self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True, sync_dist=True, batch_size=street.shape[0])
        self.train_loss.append(loss) 

        query = [x.cpu().detach().numpy() for x in street_out]
        ref = [x.cpu().detach().numpy() for x in sat_out]
        self.train_query.append(query)
        self.train_ref.append(ref)
        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x for x in self.train_loss]).mean()
        self.log('train_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        query = np.concatenate(self.train_query, axis=0)
        ref = np.concatenate(self.train_ref, axis=0)
        metrics = recall_accuracy(query, ref)
        for i in [1, 5, 10]: self.log(f'train_{i}', metrics[i], on_epoch=True, prog_bar=False, logger=True, sync_dist=True) 
        self.train_loss, self.train_query, self.train_ref = [], [], []
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        street, sat = batch['streetview'], batch['satellite']
        sat = sat.to(device)
        street = street.to(device)

        street_out, sat_out = self(street, sat)
        loss = self.mse(street_out, sat_out)

        self.log('val_loss', loss, on_step=True, prog_bar=False, logger=True, sync_dist=True, batch_size=street.shape[0])
        self.val_loss.append(loss)

        query = [x.cpu().detach().numpy() for x in street_out]
        ref = [x.cpu().detach().numpy() for x in sat_out]
        self.val_query.append(query)
        self.val_ref.append(ref)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.val_loss]).mean()
        self.log('val_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        query = np.concatenate(self.val_query, axis=0)
        ref = np.concatenate(self.val_ref, axis=0)
        metrics = recall_accuracy(query, ref)
        for i in [1, 5, 10]: self.log(f'val_{i}', metrics[i], on_epoch=True, prog_bar=False, logger=True, sync_dist=True) 

        mean_val_1_10 = torch.stack([self.trainer.callback_metrics[f'val_{i}'] for i in [1, 10]]).mean()
        self.log('val_mean', mean_val_1_10, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.val_loss, self.val_query, self.val_ref = [], [], []

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
        # streetview_keys = list(self.test_outputs['streetview'].keys()) 
        # Should be 7737 values at the end
        streetview_keys = self.test_dataset.test_order
        # print(f'ordered length: {len(streetview_keys)}')
        streetview_embeddings = [self.test_outputs['streetview'][x] for x in streetview_keys]
        satellite_keys = list(self.test_outputs['satellite'].keys())
        satellite_embeddings = [self.test_outputs['satellite'][x] for x in satellite_keys]
        
        streetview = np.concatenate(streetview_embeddings)
        satellite = np.concatenate(satellite_embeddings)

        # Calculate cosine similarity between streetview and satellite embeddings
        # print(f'input shapes: {streetview.shape}, {satellite.shape}')
        similarity = np.dot(streetview, satellite.T)
        similarity = np.argsort(similarity, axis=1)

        # print(f'\nSim Shape: {similarity.shape}\n')

        # check highest numbered folder in self.cfg.system.results_path
        results_counter = 0
        answer_file = f'{self.cfg.system.results_path}/answer.txt'
        with open(answer_file, 'w') as f:
            for idx, sim in enumerate(similarity):
                for s in sim[:10]:
                    f.write(f"{satellite_keys[s]}\t")
                f.write("\n")
                results_counter += 1
        # print(f'Number of Results: {results_counter}')
        loczip = f'{self.cfg.system.results_path}/answer.zip'
        with zipfile.ZipFile(loczip, "w", compression=zipfile.ZIP_STORED) as myzip:
            myzip.write(answer_file, arcname="answer.txt")
        # zip = zipfile.ZipFile(loczip, "w", compression=zipfile.ZIP_STORED)
        # zip.write (loczip)
        # zip.close()

        # Save config & model
        self.trainer.save_checkpoint(f"{self.cfg.system.results_path}/final_model.ckpt")
        with open(f"{self.cfg.system.results_path}/config.yaml", "w") as f:
            f.write(self.cfg.dump())
        
    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=1e-4) 
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

