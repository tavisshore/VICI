"""
Dataloaders for University1652
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import timm
from PIL import Image
from dotmap import DotMap

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lmdb_stage_keys(lmdb, stage):
    if stage == 'test':
        test_keys = []
        with open(f"src/data/query_street_name.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                test_keys.append(line.split('.')[0])
        test_keys = set(test_keys)

    image_pairs = DotMap()
    stage_keys = [x for x in list(lmdb.keys) if x.split('_')[0] == stage]
    for key in stage_keys:
        view, id = key.split('_')[1], key.split('_')[2]

        if stage != 'test':
            if id not in image_pairs:
                image_pairs[id] = DotMap(streetview=[], satellite=None)

            if view == 'street':
                image_pairs[id].streetview.append(key)
            else:
                image_pairs[id].satellite = key
        else:
            if view == 'street' and id in test_keys:
                image_pairs[id] = DotMap(streetview=key, name=id)
            elif view == 'satellite':
                image_pairs[id] = DotMap(satellite=key, name=id)

    return image_pairs, test_keys if stage == 'test' else None


class University1652_CVGL(Dataset):
    def __init__(self, cfg=None, stage: str = 'train', data_config=None, lmdb=None):
        self.cfg = cfg
        self.stage = stage
        self.root = Path(self.cfg.data.root) / stage if stage != 'val' else Path(self.cfg.data.root) / 'test'
        
        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=True if stage == 'train' else False,
                                                    #  train_crop_mode='random',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        self.lmdb = lmdb
        self.image_pairs, self.test_keys = lmdb_stage_keys(lmdb, stage)
        # self.pair_keys = [DotMap(pair=id, length=len(self.image_pairs[id].streetview)) for id in self.image_pairs]
        # For each streetview image, add to pair keys
        self.pair_keys = []
        for id in self.image_pairs:
            if stage == 'test':
                self.pair_keys.append(DotMap(pair=id, length=1))
            else:
                for i in range(len(self.image_pairs[id].streetview)):
                    self.pair_keys.append(DotMap(pair=id, street_idx=i))

    def __len__(self):
        return len(self.pair_keys)
    
    def __getitem__(self, idx):
        sample = self.pair_keys[idx]
        if self.stage == 'test':
            imgs = self.image_pairs[sample.pair]
            if isinstance(imgs.streetview, str): # Not ideal
                streetview = self.lmdb[self.image_pairs[sample.pair].streetview].convert('RGB')
                streetview = self.transform(streetview)
                return {'streetview': streetview, 'name': str(imgs.name)}
            else:
                satellite = self.lmdb[self.image_pairs[sample.pair].satellite].convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'name': str(imgs.name)}
        else:        
            streetview = self.lmdb[self.image_pairs[sample.pair].streetview[sample.street_idx]].convert('RGB')
            satellite = self.lmdb[self.image_pairs[sample.pair].satellite].convert('RGB')
            streetview = self.transform(streetview)
            satellite = self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite, 'label': sample.pair}



            
if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    from database import ImageDatabase
    cfg = CN()
    cfg.data = CN()
    cfg.data.root = '/home/shitbox/datasets/lmdb/'
    cfg.data.augment = True


    model = timm.create_model('timm/convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)

    lmdb_dataset = ImageDatabase(path=cfg.data.root)

    data = University1652_CVGL(cfg=cfg, stage='test', data_config=data_config, lmdb=lmdb_dataset)
    # item = data.__getitem__(10)




