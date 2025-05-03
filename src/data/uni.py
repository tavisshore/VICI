"""
Dataloaders for University1652
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import timm
from PIL import Image
from dotmap import DotMap
from src.data.database import ImageDatabase

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lmdb_stage_keys(lmdb, stage):
    image_pairs = DotMap()
    all_keys = lmdb.keys

    if stage == 'train':
        sat_keys = [x for x in all_keys if '_' not in x]
        street_keys = [x for x in all_keys if '_' in x]
        for sat_id in sat_keys:
            image_pairs[sat_id] = [x for x in street_keys if x.split('_')[0] == sat_id]
    elif stage =='val':
        sat_keys = [x for x in all_keys if '_' not in x]
        street_keys = [x for x in all_keys if '_' in x]
        counter = 0
        for sat_id in sat_keys:
            image_pairs[counter] = DotMap(satellite=sat_id, id=int(sat_id))
            counter += 1
        for street in street_keys:
            image_pairs[counter] = DotMap(streetview=street, id=int(street.split('_')[0]))
            counter += 1
    else:
        sat_keys = [x for x in all_keys if '_sat' in x]
        street_keys = [x for x in all_keys if '_sat' not in x]

        counter = 0
        for sat_id in sat_keys:
            image_pairs[counter] = DotMap(satellite=sat_id)
            counter += 1
        for street in street_keys:
            image_pairs[counter] = DotMap(streetview=street)
            counter += 1

    return image_pairs



def raw_stage_keys(root, cfg, stage):
    # NOTE: Quite a lot of repetition here

    image_pairs = DotMap()
    if stage == 'train':
        satellite_path = root / 'satellite'
        sub_dirs = [x.stem for x in satellite_path.iterdir() if x.is_dir()]
        for id in sub_dirs:
            streetviews = [x for x in (root / 'street' / id).iterdir() if x.is_file()]
            satellite = [x for x in (root / 'satellite' / id).iterdir() if x.is_file()][0]
            image_pairs[satellite.stem] = DotMap(satellite=satellite, streetviews=streetviews)

        return image_pairs, None
    elif stage == 'val':
        satellite_path = root / 'gallery_satellite'
        street_path = root / 'query_street'
        satellite_ids = [x.stem for x in satellite_path.iterdir() if x.is_dir()]
        street_ids = [x.stem for x in street_path.iterdir() if x.is_dir()]
        
        # querys for each satellite
        counter = 0
        for id in satellite_ids:
            sat_name = satellite_path / id / f'{id}.jpg'
            if sat_name.is_file():
                image_pairs[counter] = DotMap(satellite=sat_name, id=id)
                counter += 1
                
        for id in street_ids:
            street_names = [x for x in (street_path / id).iterdir() if x.is_file()]
            for s in street_names:
                image_pairs[counter] = DotMap(streetview=s, id=id)
                counter += 1
        
        return image_pairs, satellite_ids

    elif stage == 'test':
        counter = 0

        with open(cfg.data.query_file, "r") as f:
            for line in f.readlines():
                line = line.strip()
                id = root / 'workshop_query_street' / line
                image_pairs[counter] = DotMap(streetview=id, id=id.stem)
                counter += 1

        for id in (root / 'workshop_gallery_satellite').iterdir():
            image_pairs[counter] = DotMap(satellite=id, id=id.stem)
            counter += 1

        return image_pairs, None



class University1652_RAW(Dataset):
    def __init__(self, cfg, stage: str = 'train', data_config=None):
        self.cfg = cfg
        self.stage = stage
        self.root = Path(self.cfg.data.root) / stage if stage != 'val' else Path(self.cfg.data.root) / 'test'

        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=True if stage == 'train' else False,
                                                     scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        self.images, self.satellite_ids = raw_stage_keys(self.root, self.cfg, self.stage)
        self.pair_keys = list(self.images.keys())

    def __len__(self):
        return len(self.pair_keys)
    
    def __getitem__(self, idx):
        sample = self.pair_keys[idx]

        if self.stage == 'train':        
            if self.cfg.data.sample_equal:
                index = torch.randint(0, len(self.images[sample].streetviews), (1,)).item()
            streetview = Image.open(self.images[sample].streetviews[index]).convert('RGB')
            satellite = Image.open(self.images[sample].satellite).convert('RGB')
            streetview = self.transform(streetview)
            satellite = self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite, 'id': sample}
        
        elif self.stage == 'val':
            imgs = self.images[sample]
            if 'streetview' in imgs.keys():
                streetview = Image.open(imgs.streetview).convert('RGB')
                streetview = self.transform(streetview)
                if imgs.id in self.satellite_ids:
                    return {'streetview': streetview, 'id': int(imgs.id)}
                else: # If no GT, so return -1
                    return {'streetview': streetview, 'id': -1}
            else:
                satellite = Image.open(imgs.satellite).convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'id': int(imgs.id)}
            
        elif self.stage == 'test':
            imgs = self.images[sample]
            if 'streetview' in imgs.keys():
                streetview = Image.open(imgs.streetview).convert('RGB')
                streetview = self.transform(streetview)
                return {'streetview': streetview, 'id': str(imgs.id)}
            else:
                satellite = Image.open(imgs.satellite).convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'id': str(imgs.id)}



class University1652_LMDB(Dataset):
    def __init__(self, cfg=None, stage: str = 'train', data_config=None):
        self.cfg = cfg
        self.stage = stage
        
        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=True if stage == 'train' else False,
                                                    #  train_crop_mode='random',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        self.lmdb = ImageDatabase(path=f'{cfg.data.root}/lmdb/{stage}')
        self.images = lmdb_stage_keys(self.lmdb, stage)
        self.pair_keys = list(self.images.keys())

    def __len__(self):
        return len(self.pair_keys)
    
    def __getitem__(self, idx):
        sat_id = self.pair_keys[idx]

        if self.stage != 'train':
            img_dict = self.images[sat_id]
            if 'streetview' in img_dict.keys():
                streetview = self.lmdb[img_dict.streetview].convert('RGB')
                streetview = self.transform(streetview)
                street_name = img_dict.streetview.split('.')[0] # remove .jpg
                dic = {'streetview': streetview, 'name': street_name, 'type': 'street'}
                if 'id' in img_dict.keys():
                    dic['id'] = img_dict['id']
                return dic
            else:
                satellite = self.lmdb[img_dict.satellite].convert('RGB')
                satellite = self.transform(satellite)
                sat_name = img_dict.satellite.split('_')[0] # remove _sat.jpg
                dic = {'satellite': satellite, 'name': sat_name, 'type': 'sat'}
                if 'id' in img_dict.keys():
                    dic['id'] = img_dict['id']
                return dic
        else:        
            street_view_images = self.images[sat_id]
            index = torch.randint(0, len(street_view_images), (1,)).item()

            streetview = self.lmdb[street_view_images[index]].convert('RGB')
            satellite = self.lmdb[sat_id].convert('RGB')

            streetview = self.transform(streetview)
            satellite = self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite, 'label': sat_id}

            
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
    
    # print(f'RAW')
    # for stage in ['train', 'val', 'test']:
    #     data = University1652_RAW(cfg=cfg, stage=stage, data_config=data_config)
    #     print(f'{stage} - {data.__len__()}')


    for stage in ['train', 'val', 'test']:
        data = University1652_RAW(cfg=cfg, stage=stage, data_config=data_config)
        # item = data.__getitem__(0)
        # print(item.keys())
        print(f'{stage} - {data.__len__()}')