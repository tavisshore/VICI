import lightning.pytorch as pl
from PIL import Image
from src.models.ssl import SSL
from config.cfg import get_cfg_defaults
from torch.utils.data import Dataset, DataLoader 
from pathlib import Path
import glob
import os
import timm
import numpy as np

class U1652(Dataset):
    def __init__(self, cfg=None, data_config=None):
        self.cfg = cfg
        self.root = Path(self.cfg.data.root) / 'test'
        
        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=False,
                                                    #  train_crop_mode='random',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        self.query_img_names = []

    def __len__(self):
        return len(self.query_img_names)
    
    def __getitem__(self, idx):
        base_dir = '/work1/wshah/xzhang/data/university-1652/University-1652/test/workshop_query_street'
        img_name = self.query_img_names[idx]
        img = Image.open(os.path.join(base_dir ,img_name)).convert('RGB')
        return {'image': self.transform(img), 'name': img_name}



class Query(Dataset):
    def __init__(self, cfg=None, data_config=None):
        self.cfg = cfg
        self.root = Path(self.cfg.data.root) / 'test'
        
        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=False,
                                                    #  train_crop_mode='random',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        self.query_img_names = []
        with open(f"src/data/query_street_name.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                self.query_img_names.append(line)

    def __len__(self):
        return len(self.query_img_names)
    
    def __getitem__(self, idx):
        base_dir = '/work1/wshah/xzhang/data/university-1652/University-1652/test/workshop_query_street'
        img_name = self.query_img_names[idx]
        img = Image.open(os.path.join(base_dir ,img_name)).convert('RGB')
        return {'image': self.transform(img), 'name': img_name}

class Gallery(Dataset):
    def __init__(self, cfg=None, data_config=None):
        self.cfg = cfg
        self.root = Path(self.cfg.data.root) / 'test'
        
        self.transform = timm.data.create_transform(**data_config, 
                                                    is_training=True,
                                                    #  train_crop_mode='random',
                                                    #  scale=(0.8, 1.0),
                                                    # TODO: Add augmentations
        )

        self.gallery_img_names = glob.glob('/work1/wshah/xzhang/data/university-1652/University-1652/test/workshop_gallery_satellite/*.jpg')

    def __len__(self):
        return len(self.gallery_img_names)
    
    def __getitem__(self, idx):
        img_name = self.gallery_img_names[idx]
        img = Image.open(img_name).convert('RGB')
        img_name = img_name.split('/')[-1]
        return {'image': self.transform(img), 'name': img_name}



if __name__ == '__main__':
    model_dir = 'src/results/0'

    cfg = get_cfg_defaults()
    cfg.merge_from_file(f'{model_dir}/config.yaml')

    # model = SSL(cfg)
    model = SSL.load_from_checkpoint(os.path.join(model_dir, 'ckpts/epoch=25-val_mean=8.57.ckpt'), cfg=cfg)
    model = model.cuda()
    model.eval()

    query_dataset = Query(cfg=cfg, data_config=model.model.data_config)
    gallery_dataset = Gallery(cfg=cfg, data_config=model.model.data_config)

    query_loader = DataLoader(query_dataset, batch_size=1, num_workers=2, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=1, num_workers=2, shuffle=False)

    query_img_names = []
    with open(f"src/data/query_street_name.txt", "r") as f:
        for line in f.readlines():
            line = line.strip()
            query_img_names.append(line)

    gallery_features = []
    gallery_names = []
    for g in gallery_loader:
        gallery_img = g['image'].cuda()
        gallery_name = g['name']
        f_g = model(image=gallery_img, stage='test', branch='satellite')
        f_g = f_g.cpu().detach().numpy()
        gallery_features.append(f_g)
        gallery_names.append(gallery_name)

    query_names = []
    query_features = []
    for s in query_loader:
        query_img = s['image'].cuda()
        query_name = s['name']
        f_q = model(image=query_img, stage='test', branch='streetview')
        f_q = f_q.cpu().detach().numpy()
        query_features.append(f_q)
        query_names.append(query_name)

    # Convert lists to numpy arrays
    query_features = np.vstack(query_features)  # Shape (num_queries, feature_dim)
    gallery_features = np.vstack(gallery_features)  # Shape (num_gallery, feature_dim)

    gallery_names = np.array(gallery_names).flatten()

    # Compute cosine similarity (since features are L2 normalized, dot product is cosine similarity)
    similarity_matrix = np.dot(query_features, gallery_features.T)  # Shape (num_queries, num_gallery)

    # Get top 10 indices for each query
    top10_indices = np.argsort(-similarity_matrix, axis=1)[:, :10]  # Sort descending and take top 10

    # Write results to a text file
    output_file = "answer.txt"
    with open(output_file, "w") as f:
        for i, query_name in enumerate(query_img_names):
            top10_gallery = [gallery_names[idx].split('.')[0] for idx in top10_indices[i]]
            f.write(f"{'\t'.join(top10_gallery)}\n")

    print(f"Results saved to {output_file}")

