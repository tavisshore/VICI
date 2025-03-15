"""
Dataloaders for University1652
"""
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from dotmap import DotMap

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class University1652_CVGL(Dataset):
    def __init__(self, root: str = '/home/shitbox/datasets/University-Release', batch_size: int = 32, num_workers: int = 4, drone: bool = False, stage: str = 'train',
                 augment: bool = True):
        self.stage = stage
        self.root = Path(root) / stage if stage != 'val' else Path(root) / 'train'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aug = augment
        self.drone = drone # NOTE: Implement drone data?
        proportion = 0.9

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),            
        ])

        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomResizedCrop(256, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        sat_stem = 'workshop_gallery_satellite' if stage == 'test' else 'satellite'
        street_stem = 'workshop_query_street' if stage == 'test' else 'street'

        # If test - just dict of image + what it is
        if stage == 'test':
            self.image_sets = DotMap()
            counter = 0
            for id in (self.root / sat_stem).iterdir():
                sat_name = id.stem
                self.image_sets[counter] = DotMap(satellite=id, name=sat_name)
                counter += 1
            for id in (self.root / street_stem).iterdir():
                self.image_sets[counter] = DotMap(streetview=id)
                counter += 1
        else:
            self.satellite_path = self.root / sat_stem
            self.sub_dirs = [x.stem for x in self.satellite_path.iterdir() if x.is_dir()]
            self.image_sets = DotMap()
            counter = 0
            # Get sets of images
            for id in self.sub_dirs:
                satellite = [x for x in (self.root / sat_stem / id).iterdir() if x.is_file()]
                streetviews = [x for x in (self.root / street_stem / id).iterdir() if x.is_file()]
                for streetview in streetviews:
                    self.image_sets[counter] = DotMap(satellite=satellite[0], streetview=streetview)
                    counter += 1

            set_keys = list(self.image_sets.keys())
            if stage == 'train':
                self.image_sets = [self.image_sets[x] for x in set_keys[:int(proportion*len(set_keys))]]
            elif stage == 'val':
                self.image_sets = [self.image_sets[x] for x in set_keys[int(proportion*len(set_keys)):]]


    def __len__(self):
        return len(self.image_sets)
    
    def __getitem__(self, idx):
        sample = self.image_sets[idx]

        if self.stage == 'test':
            keys = sample.keys()
            if 'streetview' in keys:
                streetview = Image.open(sample.streetview).convert('RGB')
                streetview = self.transform(streetview)
                return {'streetview': streetview}
            else:
                satellite = Image.open(sample.satellite).convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'name': sample.name}
        else:        
            streetview = Image.open(sample.streetview).convert('RGB')
            satellite = Image.open(sample.satellite).convert('RGB')
            streetview = self.augment(streetview) if self.aug else self.transform(streetview)
            satellite = self.augment(satellite) if self.aug else self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite}



            
if __name__ == '__main__':
    data = University1652_CVGL(stage='test')
    item = data.__getitem__(0)
    
        

