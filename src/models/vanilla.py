import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights
from pytorch_metric_learning import losses
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
import zipfile

from src.data.uni import University1652_CVGL
from src.utils import recall_accuracy

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ConvNextExtractor(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        if cfg.model.size == 'tiny':
            self.map_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.map_conv.classifier[2] = nn.Identity()
            self.pov_conv = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
            self.pov_conv.classifier[2] = nn.Identity()
        elif cfg.model.size == 'base':
            self.map_conv = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            self.map_conv.classifier[2] = nn.Identity()
            self.pov_conv = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            self.pov_conv.classifier[2] = nn.Identity()
        
    def embed_map(self, map_tile: torch.Tensor) -> torch.Tensor: 
        return self.map_conv(map_tile)
    
    def embed_pov(self, pov_tile: torch.Tensor): 
        image_features = self.pov_conv(pov_tile)
        return image_features
    


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
        self.test_outputs = {'streetview': {}, 'satellite': {}}

    def train_dataloader(self):
        train_dataset = University1652_CVGL(self.cfg, stage='train')
        return DataLoader(train_dataset, batch_size=16, num_workers=4, shuffle=True)

    def val_dataloader(self):
        val_dataset = University1652_CVGL(self.cfg, stage='val')
        return DataLoader(val_dataset, batch_size=16, num_workers=4, shuffle=False)
    
    def test_dataloader(self):
        self.test_dataset = University1652_CVGL(self.cfg, stage='test')
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4, shuffle=False)
    
    def forward(self, street: torch.Tensor = None, sat: torch.Tensor = None, image: torch.Tensor = None, branch: str = 'streetview', stage: str = 'train'):
        if stage == 'test':
            if branch == 'streetview':
                x = self.model.embed_pov(image)
            else:
                x = self.model.embed_map(image)
            x = F.normalize(x, p=2, dim=1)
            return x
    
        street_out = self.model.embed_map(street)
        sat_out = self.model.embed_pov(sat)
        street_out = F.normalize(street_out, p=2, dim=1)
        sat_out = F.normalize(sat_out, p=2, dim=1)
        return street_out, sat_out

    def select_triplets(self, street: torch.Tensor, sat: torch.Tensor):
        embeddings = torch.cat((street.float(), sat.float()), dim=0)
        street_len = street.shape[0]

        if self.cfg.model.selection == 'feat':
            # Select triplets based on feature similarity - most dissimilar sat is negative
            similarity = torch.nn.functional.cosine_similarity(street.unsqueeze(1), sat.unsqueeze(0), dim=2)
            
            # Get top 3 most dissimilar (lowest similarity) indices
            negatives = torch.topk(similarity, 3, largest=False, dim=1).indices 

            anchors, positives, ns = [], [], []
            for idx, neg in enumerate(negatives):
                # remove self index
                neg = neg[neg != idx]
                neg = neg.tolist()
                for n in neg:
                    anchors.append(idx)
                    positives.append(idx+street_len)
                    ns.append(n+street_len)

            anchors = torch.tensor(anchors, device=device)
            positives = torch.tensor(positives, device=device)
            negatives = torch.tensor(ns, device=device)
            return embeddings, anchors, positives, negatives
        else:
            embeddings = torch.cat((street.float(), sat.float()), dim=0)
            emb_length = street.shape[0]
            anchors = torch.arange(0, emb_length)
            positives = torch.arange(emb_length, emb_length*2)
            negatives = torch.add(torch.randint(0, emb_length, (emb_length,)), emb_length)#.repeat_interleave(self.hparams['args'].walk)
            while torch.any(negatives == torch.arange(emb_length, emb_length*2)):
                negatives = torch.add(torch.randint(0, emb_length, (emb_length,)), emb_length)
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            return embeddings, anchors, positives, negatives

    def training_step(self, batch, batch_idx):
        street, sat = batch['streetview'], batch['satellite']
        sat = sat.to(device)
        street = street.to(device)

        street_out, sat_out = self(street, sat)

        # Selects negatives based on feature similarity
        embs, anchors, positives, negatives = self.select_triplets(street_out, sat_out)

        loss = self.loss_func(embs, indices_tuple=(anchors, positives, negatives))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
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
        streetview_keys = list(self.test_outputs['streetview'].keys())

        streetview_embeddings = [self.test_outputs['streetview'][x] for x in streetview_keys]
        satellite_keys = list(self.test_outputs['satellite'].keys())
        satellite_embeddings = [self.test_outputs['satellite'][x] for x in satellite_keys]
        
        streetview = np.concatenate(streetview_embeddings)
        satellite = np.concatenate(satellite_embeddings)

        # Calculate cosine similarity between streetview and satellite embeddings
        similarity = np.dot(streetview, satellite.T)
        similarity = np.argsort(similarity, axis=1)
        answer_file = f'{self.cfg.system.path}/answer.txt'
        with open(answer_file, 'w') as f:
            for idx, sim in enumerate(similarity):
                for s in sim[:10]:
                    f.write(f"{satellite_keys[s]}\t")
                f.write("\n")
        
        loczip = f'{self.cfg.system.path}/answer.zip'
        zip = zipfile.ZipFile(loczip, "w")
        zip.write (loczip)
        zip.close()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(params=self.parameters(), lr=1e-4) #if self.hparams.gnn else torch.optim.AdamW(params=self.further_encoder.parameters(), lr=self.args.lr)
        sch = ReduceLROnPlateau(optimizer=opt, mode='min', factor=0.66, patience=2, verbose=True)
        return [opt], [{"scheduler": sch, "interval": "epoch", "monitor": "train_epoch_loss"}]

