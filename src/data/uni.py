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



class University1652_CVGL(Dataset):
    def __init__(self, cfg, stage: str = 'train', data_config=None):
        self.cfg = cfg
        self.stage = stage
        self.root = Path(self.cfg.data.root) / stage if stage != 'val' else Path(self.cfg.data.root) / 'test'
        
        self.transform = timm.data.create_transform(**data_config, is_training=True if stage == 'train' else False)

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

                    if not self.cfg.data.sample_equal: # Every possible image pair is a sample
                        self.pair_keys.append(DotMap(pair=id, index=i_s))
                if self.cfg.data.sample_equal: # Only one image pair per satellite reference - randomly select streetview at runtime
                    self.pair_keys.append(DotMap(pair=id, index=list(range(len(streetviews)))))
        elif stage == 'val':
            self.satellite_path = self.root / 'query_satellite'
            self.sub_dirs = [x.stem for x in self.satellite_path.iterdir() if x.is_dir()]

            self.pair_keys = []
            for id in self.sub_dirs: # querys for each satellite
                # 1 satellite, varying streetview - add sampling option to not bias
                streetviews = [x for x in (self.root / 'query_street' / id).iterdir() if x.is_file()]
                satellite = [x for x in (self.root / 'query_satellite' / id).iterdir() if x.is_file()]
                self.image_pairs[id] = DotMap()
                sat_counter += 1

                for i_s, s in enumerate(streetviews):
                    self.image_pairs[id][i_s] = DotMap(streetview=s, satellite=satellite[0], pair=id, idx=i_s)
                    self.pair_keys.append(DotMap(pair=id, index=i_s))
                    street_counter += 1
        elif stage == 'test':
            counter = 0
            with open(f"{self.cfg.data.root}/test/query_street_name.txt", "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    id = self.root / 'workshop_query_street' / line
                    self.image_pairs[counter] = DotMap(streetview=id, name=id.stem)
                    counter += 1
            for id in (self.root / 'workshop_gallery_satellite').iterdir():
                self.image_pairs[counter] = DotMap(satellite=id, name=id.stem)
                counter += 1
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
        else:        
            if self.cfg.data.sample_equal:
                sample.index = torch.randint(0, len(self.image_pairs[sample.pair]), (1,)).item()
            imgs = self.image_pairs[sample.pair][sample.index]
            streetview = Image.open(imgs.streetview).convert('RGB')
            satellite = Image.open(imgs.satellite).convert('RGB')
            streetview = self.transform(streetview)
            satellite = self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite, 'label': sample.pair}



            
if __name__ == '__main__':
    from yacs.config import CfgNode as CN
    cfg = CN()
    cfg.data = CN()
    cfg.data.root = '/home/shitbox/datasets/University-Release/'
    cfg.data.augment = True
    cfg.data.sample_equal = True
    data = University1652_CVGL(cfg=cfg, stage='test')
    item = data.__getitem__(0)
    
    
        

