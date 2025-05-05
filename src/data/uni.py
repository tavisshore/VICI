"""
Dataloaders for University1652
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import timm
from PIL import Image
from dotmap import DotMap
# from src.data.database import ImageDatabase
from database import ImageDatabase

import random 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lmdb_stage_keys(cfg, lmdb, stage, val_prop=0.01):
    image_pairs = DotMap()
    all_keys = lmdb.keys

    if stage == 'train' or stage == 'val':
        street_keys = [x for x in all_keys if 'street_' in x]
        sat_keys = [x for x in all_keys if 'satellite_' in x]
        drone_keys = [x for x in all_keys if 'drone_' in x]
        google_keys = [x for x in all_keys if 'google_' in x]

        for sat_id in sat_keys:
            image_pairs[sat_id] = [x for x in street_keys if x.split('_')[1] == sat_id.split('_')[1]]

            # Use additional non-satellite images
            if cfg.data.use_drone:
                image_pairs[sat_id] += [x for x in drone_keys if x.split('_')[1] == sat_id.split('_')[1]]

            if cfg.data.use_google:
                image_pairs[sat_id] += [x for x in google_keys if x.split('_')[1] == sat_id.split('_')[1]]

        img_keys = list(image_pairs.keys())
        if stage == 'val':
            img_keys = img_keys[int((1-val_prop)*len(img_keys)):]
        else:
            img_keys = img_keys[:int((1-val_prop)*len(img_keys))]                

        new_dict = DotMap()
        for k in img_keys:
            new_dict[k] = image_pairs[k]
        image_pairs = new_dict

    else:
        sat_keys = [x for x in all_keys if 'satellite_' in x]
        street_keys = [x for x in all_keys if 'street_' in x]

        counter = 0
        for sat_id in sat_keys:
            image_pairs[counter] = DotMap(satellite=sat_id)
            counter += 1
        for street in street_keys:
            image_pairs[counter] = DotMap(streetview=street)
            counter += 1

    return image_pairs


class University1652_LMDB(Dataset):
    def __init__(self, cfg=None, stage: str = 'train', data_config=None):
        self.cfg = cfg
        self.stage = stage
        self.data_stage = 'train' if stage == 'val' else stage
        
        # self.transform = timm.data.create_transform(**data_config, 
        #                                             is_training=True if stage == 'train' else False,
        #                                             #  train_crop_mode='random',
        #                                             #  scale=(0.8, 1.0),
        #                                             # TODO: Add augmentations
        # )
        
        self.lmdb = ImageDatabase(path=f'{cfg.data.root}/lmdb/{self.data_stage}')
        self.images = lmdb_stage_keys(cfg, self.lmdb, stage)
        self.pair_keys = list(self.images.keys())

    def __len__(self):
        return len(self.pair_keys)
    
    def __getitem__(self, idx):
        sat_id = self.pair_keys[idx]
        # print(f'sat_id: {sat_id}')
        if self.stage == 'test':
            img_dict = self.images[sat_id]

            if 'streetview' in img_dict.keys():
                streetview = self.lmdb[img_dict.streetview].convert('RGB')
                # streetview = self.transform(streetview)
                street_name = img_dict.streetview.split('.')[0] # remove .jpg
                dic = {'streetview': streetview, 'label': street_name, 'type': 'street'}
                if 'id' in img_dict.keys():
                    dic['id'] = img_dict['id']
                return dic
            else:
                satellite = self.lmdb[img_dict.satellite].convert('RGB')
                # satellite = self.transform(satellite)
                sat_name = img_dict.satellite.split('_')[1]
                dic = {'satellite': satellite, 'label': sat_name, 'type': 'sat'}
                if 'id' in img_dict.keys():
                    dic['id'] = img_dict['id']
                return dic
        else:        
            non_sat_images = self.images[sat_id]
            index = torch.randint(0, len(non_sat_images), (1,)).item()
            streetview = self.lmdb[non_sat_images[index]].convert('RGB')
            satellite = self.lmdb[sat_id].convert('RGB')
            sat_name = sat_id.split('_')[1]
            # streetview = self.transform(streetview)
            # satellite = self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite, 'label': sat_name}

            
if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.data = CN()
    cfg.data.root = '/scratch/datasets/University/lmdb'

    cfg.data.augment = True


    model = timm.create_model('timm/convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True, num_classes=0)
    data_config = timm.data.resolve_model_data_config(model)

    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.data = CN()
    cfg.data.root = '/scratch/datasets/University/'
    cfg.data.sample_equal = True
    cfg.data.query_file = '/scratch/projects/challenge/src/data/query_street_name.txt'
    cfg.data.use_drone = False
    cfg.data.use_google = True
    
    # print(f'RAW')
    # for stage in ['train', 'val', 'test']:
    #     data = University1652_RAW(cfg=cfg, stage=stage, data_config=data_config)
    #     print(f'{stage} - {data.__len__()}')


    for stage in ['train', 'val', 'test']:
        data = University1652_LMDB(cfg=cfg, stage=stage, data_config=data_config)
        # item = data.__getitem__(0)
        # print(item)
        print(f'{stage} - {data.__len__()}')