from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights
from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn import functional as F
import zipfile
from dotmap import DotMap

from src.data.uni import University1652_CVGL
from src.utils import recall_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvNextExtractor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model.size == 'tiny':
            self.street_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.street_conv.classifier[2] = nn.Identity()
            self.sat_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.sat_conv.classifier[2] = nn.Identity()
        elif cfg.model.size == 'base':
            self.street_conv = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            self.street_conv.classifier[2] = nn.Identity()
            self.sat_conv = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            self.sat_conv.classifier[2] = nn.Identity()
        
    def embed_street(self, pov_tile: torch.Tensor): 
        return self.street_conv(pov_tile)
    
    def embed_sat(self, map_tile: torch.Tensor) -> torch.Tensor: 
        return self.sat_conv(map_tile)
    

    


class Vanilla(pl.LightningModule):
    def __init__(self, cfg):
        super(Vanilla, self).__init__()
        self.cfg = cfg

        self.model = ConvNextExtractor(cfg)
        self.model.to(device)

        self.loss_func = losses.NTXentLoss()
        self.mse = nn.MSELoss()
        
        self.train_loss, self.val_loss, self.test_loss = [], [], []
        self.train_query, self.train_ref = [], []
        self.val_query, self.val_ref = [], []
        self.test_outputs = DotMap(streetview=DotMap(), satellite=DotMap())

    def train_dataloader(self):
        train_dataset = University1652_CVGL(self.cfg, stage='train')
        return DataLoader(train_dataset, batch_size=self.cfg.system.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        val_dataset = University1652_CVGL(self.cfg, stage='val')
        return DataLoader(val_dataset, batch_size=self.cfg.system.batch_size, num_workers=4, shuffle=False)
    
    def test_dataloader(self):
        self.test_dataset = University1652_CVGL(self.cfg, stage='test')
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
        """
        Triplet sampling methods - update once dataset sampler implemented
        """
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
        loss = self.loss_func(embs, indices_tuple=(anchors, positives, negatives))

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=street.shape[0])
        self.train_loss.append(loss) 

        query = [x.cpu().detach().numpy() for x in street_out]
        ref = [x.cpu().detach().numpy() for x in sat_out]
        self.train_query.append(query)
        self.train_ref.append(ref)

        return loss
    
    def on_train_epoch_end(self):
        avg_loss = torch.stack([x for x in self.train_loss]).mean()
        self.log('train_epoch_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.train_loss = []

        query = np.concatenate(self.train_query, axis=0)
        ref = np.concatenate(self.train_ref, axis=0)
        metrics = recall_accuracy(query, ref)
        self.log('train_1', metrics[1], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_5', metrics[5], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_10', metrics[10], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.train_query, self.train_ref = [], []
        return avg_loss
    
    def validation_step(self, batch, batch_idx):
        street, sat = batch['streetview'], batch['satellite']
        sat = sat.to(device)
        street = street.to(device)

        street_out, sat_out = self(street, sat)
        loss = self.mse(street_out, sat_out)

        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=street.shape[0])
        self.val_loss.append(loss)

        query = [x.cpu().detach().numpy() for x in street_out]
        ref = [x.cpu().detach().numpy() for x in sat_out]
        self.val_query.append(query)
        self.val_ref.append(ref)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x for x in self.val_loss]).mean()
        self.log('val_epoch_loss', avg_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_loss = []

        query = np.concatenate(self.val_query, axis=0)
        ref = np.concatenate(self.val_ref, axis=0)
        metrics = recall_accuracy(query, ref)
        self.log('val_1', metrics[1], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_5', metrics[5], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_10', metrics[10], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.val_query, self.val_ref = [], []
        return
    
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
        # streetview_keys = list(self.test_outputs['streetview'].keys()) # Orders test outputs by dataset.test_order
        # Should be 7737 values at the end
        streetview_keys = self.test_dataset.test_order
        print(f'ordered length: {len(streetview_keys)}')
        streetview_embeddings = [self.test_outputs['streetview'][x] for x in streetview_keys]
        satellite_keys = list(self.test_outputs['satellite'].keys())
        satellite_embeddings = [self.test_outputs['satellite'][x] for x in satellite_keys]
        
        streetview = np.concatenate(streetview_embeddings)
        satellite = np.concatenate(satellite_embeddings)

        # Calculate cosine similarity between streetview and satellite embeddings
        print(f'input shapes: {streetview.shape}, {satellite.shape}')
        similarity = np.dot(streetview, satellite.T)
        similarity = np.argsort(similarity, axis=1)

        print(f'\nSim Shape: {similarity.shape}\n')

        # check highest numbered folder in self.cfg.system.results_path
        results_counter = 0
        folder = [f.name for f in Path(self.cfg.system.results_path).iterdir() if f.is_dir()]
        folder = [int(f) for f in folder if f.isdigit()]
        folder = max(folder) + 1 if folder else 0
        folder = f'{self.cfg.system.results_path}/{folder}'
        Path(folder).mkdir(parents=True, exist_ok=True)
        answer_file = f'{folder}/answer.txt'
        with open(answer_file, 'w') as f:
            for idx, sim in enumerate(similarity):
                for s in sim[:10]:
                    f.write(f"{satellite_keys[s]}\t")
                f.write("\n")
                results_counter += 1
        print(f'Number of Results: {results_counter}')
        loczip = f'{folder}/answer.zip'
        zip = zipfile.ZipFile(loczip, "w", compression=zipfile.ZIP_STORED)
        zip.write (loczip)
        zip.close()
        return folder

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=1e-4) #if self.hparams.gnn else torch.optim.AdamW(params=self.further_encoder.parameters(), lr=self.args.lr)
        # sch = ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.5, patience=2, verbose=True)
        sch = StepLR(optimizer=opt, step_size=40, gamma=0.5, verbose=True)
        return [opt], [{"scheduler": sch, "interval": "epoch", "monitor": "train_epoch_loss"}]

