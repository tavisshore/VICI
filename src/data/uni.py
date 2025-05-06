"""
Dataloaders for University1652
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import timm
from dotmap import DotMap
if __name__ == '__main__':
    from database import ImageDatabase
else:
    from src.data.database import ImageDatabase
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lmdb_stage_keys(cfg, lmdb, stage):
    image_pairs = DotMap()

    if stage == 'train' or stage == 'val':
        all_keys = lmdb.keys
        street_keys = set([x for x in all_keys if 'street_' in x])
        sat_keys = set([x for x in all_keys if 'satellite_' in x])
        drone_keys = set([x for x in all_keys if 'drone_' in x])
        google_keys = set([x for x in all_keys if 'google_' in x])

        for sat_id in sat_keys:
            image_pairs[sat_id] = DotMap()
            image_pairs[sat_id].streetview = [x for x in street_keys if x.split('_')[1] == sat_id.split('_')[1]]

            # Use additional non-satellite images
            if cfg.data.use_google:
                image_pairs[sat_id].streetview += [x for x in google_keys if x.split('_')[1] == sat_id.split('_')[1]]

            if cfg.data.use_drone:
                image_pairs[sat_id].drone = [x for x in drone_keys if x.split('_')[1] == sat_id.split('_')[1]]

        img_keys = list(image_pairs.keys())
        if stage == 'val':
            img_keys = img_keys[int((1-cfg.data.val_prop)*len(img_keys)):]
        else:
            img_keys = img_keys[:int((1-cfg.data.val_prop)*len(img_keys))]                

        new_dict = DotMap()
        for k in img_keys:
            new_dict[k] = image_pairs[k]
        image_pairs = new_dict
    else:
        counter = 0
        with open(cfg.data.query_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                id = Path(cfg.data.root) / 'test/workshop_query_street' / line
                image_pairs[counter] = DotMap(streetview=id, label=id.stem)
                counter += 1

        for id in (Path(cfg.data.root) / 'test/workshop_gallery_satellite').iterdir():
            image_pairs[counter] = DotMap(satellite=id, label=id.stem)
            counter += 1

    return image_pairs


class University1652_LMDB(Dataset):
    def __init__(self, cfg=None, stage: str = 'train', data_config=None):
        self.cfg = cfg
        self.stage = stage
        self.data_stage = 'train' if stage == 'val' else stage
        
        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=True if stage == 'train' else False,
                                                    #  train_crop_mode='rrc',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )
        
        self.lmdb = ImageDatabase(path=f'{cfg.data.root}/lmdb/{self.data_stage}') if stage != 'test' else None
        self.images = lmdb_stage_keys(cfg, self.lmdb, stage)
        self.pair_keys = list(self.images.keys())

    def __len__(self):
        return len(self.pair_keys)
    
    def __getitem__(self, idx):
        sat_id = self.pair_keys[idx]

        if self.stage == 'test':
            img_dict = self.images[sat_id]
            keys = img_dict.keys()
            if 'streetview' in keys:
                streetview = Image.open(img_dict.streetview).convert('RGB')
                streetview = self.transform(streetview)
                return {'streetview': streetview, 'label': str(img_dict.label)}
            else:
                satellite = Image.open(img_dict.satellite).convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'label': str(img_dict.label)}
        else:        
            non_sat_images = self.images[sat_id]
            index = torch.randint(0, len(non_sat_images.streetview), (1,)).item()
            streetview = self.lmdb[non_sat_images.streetview[index]].convert('RGB')
            satellite = self.lmdb[sat_id].convert('RGB')
            sat_name = sat_id.split('_')[1]

            if self.cfg.data.use_drone and 'drone' in non_sat_images.keys():
                index = torch.randint(0, len(non_sat_images.drone), (1,)).item()
                drone = self.lmdb[non_sat_images.drone[index]].convert('RGB')
                drone = self.transform(drone)
            else: drone = torch.tensor([])
            streetview = self.transform(streetview)
            satellite = self.transform(satellite)
            return {'streetview': streetview, 'drone': drone, 'satellite': satellite, 'label': sat_name}

            
if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.data = CN()
    cfg.data.root = '/scratch/datasets/University/lmdb'

    cfg.data.augment = True


    model = timm.create_model('timm/convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)

    from yacs.config import CfgNode as CN
    import random

    cfg = CN()
    cfg.data = CN()
    cfg.data.root = '/scratch/datasets/University/'
    cfg.data.sample_equal = True
    cfg.data.query_file = '/scratch/projects/challenge/src/data/query_street_name.txt'
    cfg.data.use_drone = True
    cfg.data.use_google = True
    cfg.data.val_prop = 0.0
    
    for stage in ['train']:
        data = University1652_LMDB(cfg=cfg, stage=stage, data_config=data_config)
        item = data.__getitem__(random.randint(0, len(data)))
        print(item)
        # item['streetview'].save('streetview.jpg') 
        # item['satellite'].save('satellite.jpg')
        # item['drone'].save('drone.jpg')
        # print(f'{stage} - {data.__len__()}')