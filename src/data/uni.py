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
    def __init__(self, root: str = '/home/shitbox/datasets/University-Release/', batch_size: int = 32, num_workers: int = 4, drone: bool = False, stage: str = 'train',
                 augment: bool = True, samearea: bool = False):
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
        self.cross_views = DotMap()

        if stage == 'test':
            counter = 0
            for id in (self.root / sat_stem).iterdir():
                self.cross_views[counter] = DotMap(satellite=id, name=id.stem)
                counter += 1
            for id in (self.root / street_stem).iterdir():
                self.cross_views[counter] = DotMap(streetview=id, name=id.stem)
                counter += 1
            self.image_sets = list(self.cross_views.keys())
            my_file = open("/home/shitbox/datasets/University-Release/test/query_street_name.txt", "r") 
            data = my_file.read() 
            self.test_order = data.split("\n") 
            self.test_order = [x.split('.')[0] for x in self.test_order][:-1]
            my_file.close() 
        else:
            self.satellite_path = self.root / sat_stem
            self.sub_dirs = [x.stem for x in self.satellite_path.iterdir() if x.is_dir()]
            self.image_sets = []

            # TODO: change to using id - with sub dicts - allowing for cross-area vs same-area
            # Get sets of images
            for id in self.sub_dirs:
                satellite = [x for x in (self.root / sat_stem / id).iterdir() if x.is_file()]
                streetviews = [x for x in (self.root / street_stem / id).iterdir() if x.is_file()]
                self.cross_views[id] = DotMap(satellite=satellite, streetview=streetviews)
            set_keys = list(self.cross_views.keys())

            # Validation references same as train or not - split before or after - better way?
            if samearea:
                for key in set_keys:
                    sat = str(self.cross_views[key].satellite[0])
                    street = self.cross_views[key].streetview
                    for s in street:
                        self.image_sets.append(DotMap(satellite=sat, streetview=s))
                if stage == 'train':
                    self.image_sets = self.image_sets[:int(proportion*len(self.image_sets))]
                elif stage == 'val':
                    self.image_sets = self.image_sets[int(proportion*len(self.image_sets)):]
            else:
                if stage == 'train':
                    keys = set_keys[:int(proportion*len(set_keys))]
                elif stage == 'val':
                    keys = set_keys[int(proportion*len(set_keys)):]
                for key in keys:
                    sat = str(self.cross_views[key].satellite[0])
                    street = self.cross_views[key].streetview
                    for s in street:
                        self.image_sets.append(DotMap(satellite=sat, streetview=s))

    def __len__(self):
        return len(self.image_sets)
    
    def __getitem__(self, idx):
        sample = self.image_sets[idx]

        if self.stage == 'test':
            sample = self.cross_views[sample]
            keys = sample.keys()
            if 'streetview' in keys:
                streetview = Image.open(sample.streetview).convert('RGB')
                streetview = self.transform(streetview)
                return {'streetview': streetview, 'name': str(sample.name)}
            else:
                satellite = Image.open(sample.satellite).convert('RGB')
                satellite = self.transform(satellite)
                return {'satellite': satellite, 'name': str(sample.name)}
        else:        
            streetview = Image.open(sample.streetview).convert('RGB')
            satellite = Image.open(sample.satellite).convert('RGB')
            streetview = self.augment(streetview) if self.aug else self.transform(streetview)
            satellite = self.augment(satellite) if self.aug else self.transform(satellite)
            return {'streetview': streetview, 'satellite': satellite}



            
if __name__ == '__main__':
    data = University1652_CVGL(stage='test')
    item = data.__getitem__(0)
    
        

