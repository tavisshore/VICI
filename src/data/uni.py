"""
Dataloaders for University1652
"""
from pathlib import Path
import torch
from torch.utils.data import Dataset
import timm
from PIL import Image
from dotmap import DotMap
from database import ImageDatabase

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def lmdb_stage_keys(lmdb, stage):
    image_pairs = DotMap()

    all_keys = lmdb.keys

    if stage != 'test':
        sat_keys = [x for x in all_keys if '_' not in x]
        street_keys = [x for x in all_keys if '_' in x]
        for sat_id in sat_keys:
            image_pairs[sat_id] = [x for x in street_keys if x.split('_')[0] == sat_id]

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


class University1652_RAW(Dataset):
    def __init__(self, cfg, stage: str = 'train', data_config=None):
        self.cfg = cfg
        self.stage = stage
        self.root = Path(self.cfg.data.root) / stage if stage != 'val' else Path(self.cfg.data.root) / 'test'

        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=True if stage == 'train' else False,
                                                    #  train_crop_mode='random',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        self.image_pairs = DotMap()
        sat_counter, street_counter = 0, 0

        # NOTE: Quite a lot of repetition here
        if stage == 'train':
            self.satellite_path = self.root / 'satellite'
            self.sub_dirs = [x.stem for x in self.satellite_path.iterdir() if x.is_dir()]
            
            # Validation references same as train or not - split before or after - better way?
            self.pair_keys = []
            for id in self.sub_dirs:
                # 1 satellite, varying streetview - add sampling option to not bias
                streetviews = [x for x in (self.root / 'street' / id).iterdir() if x.is_file()]
                satellite = [x for x in (self.root / 'satellite' / id).iterdir() if x.is_file()]
                self.image_pairs[id] = DotMap()
                sat_counter += 1
                for i_s, s in enumerate(streetviews):
                    self.image_pairs[id][i_s] = DotMap(streetview=s, satellite=satellite[0], pair=id, idx=i_s)
                    street_counter += 1

                # TODO: This dodgy bit of code simplifies sampling - evaluate now.
                    if not self.cfg.data.sample_equal: # Every possible image pair is a sample
                        self.pair_keys.append(DotMap(pair=id, index=i_s))
                if self.cfg.data.sample_equal: # Only one image pair per satellite reference - randomly select streetview at runtime
                    self.pair_keys.append(DotMap(pair=id, index=list(range(len(streetviews)))))
        elif stage == 'val':
            self.satellite_path = self.root / 'gallery_satellite'
            self.street_path = self.root / 'query_street'
            self.satellite_ids = [x.stem for x in self.satellite_path.iterdir() if x.is_dir()]
            self.street_ids = [x.stem for x in self.street_path.iterdir() if x.is_dir()]

            self.pair_keys = []
            # querys for each satellite
            counter = 0
            for id in self.satellite_ids:
                sat_name = self.satellite_path / id / f'{id}.jpg'
                if sat_name.is_file():
                    self.image_pairs[counter] = DotMap(satellite=sat_name, name=id)
                    counter += 1
                    
            for id in self.street_ids:
                street_names = [x for x in (self.street_path / id).iterdir() if x.is_file()]
                for s in street_names:
                    self.image_pairs[counter] = DotMap(streetview=s, name=id)
                    counter += 1

            self.pair_keys = list(self.image_pairs.keys())

        elif stage == 'test':
            counter = 0

            with open(self.cfg.data.query_file, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    id = self.root / 'workshop_query_street' / line
                    self.image_pairs[counter] = DotMap(streetview=id, name=id.stem)
                    counter += 1
                    street_counter += 1

            for id in (self.root / 'workshop_gallery_satellite').iterdir():
                self.image_pairs[counter] = DotMap(satellite=id, name=id.stem)
                counter += 1
                sat_counter += 1
            self.pair_keys = list(self.image_pairs.keys())

    def __len__(self):
        return len(self.pair_keys)
    
    def __getitem__(self, idx):
        sample = self.pair_keys[idx]

        if self.stage == 'test':
            imgs = self.image_pairs[sample]
            keys = imgs.keys()
            if 'streetview' in keys:
                streetview = Image.open(imgs.streetview).convert('RGB')
                streetview = self.transform(streetview)
                return {'streetview': streetview, 'name': str(imgs.name)}
            else:
                satellite = Image.open(imgs.satellite).convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'name': str(imgs.name)}
        elif self.stage == 'train':        
            if self.cfg.data.sample_equal:
                sample.index = torch.randint(0, len(self.image_pairs[sample.pair]), (1,)).item()
            imgs = self.image_pairs[sample.pair][sample.index]
            streetview = Image.open(imgs.streetview).convert('RGB')
            satellite = Image.open(imgs.satellite).convert('RGB')
            streetview = self.transform(streetview)
            satellite = self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite, 'label': sample.pair}

        else:
            imgs = self.image_pairs[sample]
            keys = imgs.keys()
            if 'streetview' in keys:
                streetview = Image.open(imgs.streetview).convert('RGB')
                streetview = self.transform(streetview)
                if imgs.name in self.satellite_ids:
                    return {'streetview': streetview, 'name': int(imgs.name)}
                else: # If no GT, so return -1
                    return {'streetview': streetview, 'name': -1}
            else:
                satellite = Image.open(imgs.satellite).convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'name': int(imgs.name)}

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

        if self.stage == 'test':
            img_dict = self.images[sat_id]
            if 'streetview' in img_dict.keys():
                streetview = self.lmdb[img_dict.streetview].convert('RGB')
                streetview = self.transform(streetview)
                street_name = img_dict.streetview.split('.')[0] # remove _sat.jpg
                return {'streetview': streetview, 'name': street_name, 'type': 'street'}
            else:
                satellite = self.lmdb[img_dict.satellite].convert('RGB')
                satellite = self.transform(satellite)
                sat_name = img_dict.satellite.split('_')[0] # remove .jpg
                return {'satellite': satellite, 'name': sat_name, 'type': 'sat'}
        else:        
            street_view_images = self.images[sat_id]
            index = torch.randint(0, len(street_view_images), (1,)).item()

            streetview = self.lmdb[street_view_images[index]].convert('RGB')
            satellite = self.lmdb[sat_id].convert('RGB')

            streetview = self.transform(streetview)
            satellite = self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite, 'sat_id': sat_id}

            
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
    cfg.data.sample_equal = False
    cfg.data.query_file = '/scratch/projects/challenge/src/data/query_street_name.txt'
    
    # print(f'RAW')


    # for stage in ['train', 'val', 'test']:
    #     data = University1652_RAW(cfg=cfg, stage=stage, data_config=data_config)
    #     print(f'{stage} - {data.__len__()}')

    # print(f'LMDB')

    for stage in ['test']:
        data = University1652_LMDB(cfg=cfg, stage=stage, data_config=data_config)
