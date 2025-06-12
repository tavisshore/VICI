"""
Dataloaders for University1652
"""
import torch
import timm
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dotmap import DotMap
import random
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OverSampler(torch.utils.data.Sampler):
    """
    Sample cross-views from dataset at equal rate per satellite reference
    """
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.indices = list(range(len(data_source)))

        sample_weights = {}
        for i in self.data_source.image_pairs.keys():
            sample_weights[i] = len(self.data_source.image_pairs[i])
        max_weight = max(sample_weights.values())
        for i in sample_weights.keys():
            sample_weights[i] = max_weight - sample_weights[i]
        # self.indices = [idx for idx in self.indices for _ in range(sample_weights[self.data_source.pair_keys[idx].pair])
        print(sample_weights)
      
        # self.indices = [idx for idx in self.indices for _ in range(sample_weights[self.data_source.pair_keys[idx].pair])

        if self.shuffle:
            torch.manual_seed(0)
            torch.randperm(len(data_source))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.data_source)
    


class University1652_CVGL(Dataset):
    def __init__(self, cfg, stage: str = 'train', data_config=None):
        self.cfg = cfg
        self.stage = stage
        self.root = Path(self.cfg.data.root) / stage if stage != 'val' else Path(self.cfg.data.root) / 'test'
        
        # if stage == "train":
        #     data_config['re_prob'] = 0.2
        #     data_config['grayscale_prob'] = 0.2
        #     data_config['gaussian_blur_prob'] = 0.2
            

        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=True if stage == 'train' else False,
                                                    #  train_crop_mode='random',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        # print("==================================")
        # print(stage)
        # print(self.transform)
        # print("==================================")

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

            # ADD GOOGLE AND MAYBE DRONE IMAGES TO SET
            if self.cfg.data.google_input:
                pass

            if self.cfg.data.drone_input:
                pass


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

            # Randomly replace satellite with drone image
            replace_satellite = random.random() < self.cfg.data.drone_image_rate
            if self.cfg.data.include_drone and replace_satellite:
                # I hardcoded the path to drone images, assuming they are in the same directory structure as satellite images.
                # This is a bit hacky, but it works for the dataset structure.
                # Please change this accordingly for LMDB.
                drone_path = str(imgs.satellite).replace('satellite', 'drone').split('/')[:-1]
                drone_path = '/'.join(drone_path)
                # TODO: Here we may consider only using first 20 or 30 drone images. They are usually the high altitude ones.
                drone_img_list = [x for x in Path(drone_path).iterdir() if x.is_file()]
                random_drone_img = random.choice(drone_img_list)
                satellite = Image.open(random_drone_img).convert('RGB')
                
            else:
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



            
if __name__ == '__main__':
    # from yacs.config import CfgNode as CN
    # cfg = CN()
    # cfg.data = CN()
    # cfg.data.root = '/scratch/datasets/University/'
    # cfg.data.sample_equal = False
    # cfg.data.query_file = '/scratch/projects/challenge/src/data/query_street_name.txt'

    # for stage in ['test']:
    #     data = University1652_CVGL(cfg=cfg, stage=stage)
    #     print(f'{stage} - {data.__len__()}')

    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.data = CN()
    cfg.data.root = '/work1/wshah/xzhang/data/university-1652/University-1652'
    cfg.data.sample_equal = True
    data = University1652_CVGL(cfg=cfg, stage='train', data_config={'input_size': 224})

    i = 0
    while i < 10:
        i += 1
        item = data.__getitem__(i)
        
        print('len: ', len(data))

        try:
            print(item['streetview'].shape)
        except:
            print(item['satellite'].shape)

        # print(type(item['name']))