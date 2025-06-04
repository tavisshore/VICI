import torch
import timm
import numpy as np
from dataclasses import dataclass
import os
from openai import OpenAI
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimmModel(nn.Module):
    def __init__(self, model_name, pretrained=True, img_size=383):
        super(TimmModel, self).__init__()
        self.img_size = img_size
        if "vit" in model_name:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, img_size=img_size)
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img):
        return self.model(img)

@dataclass
class Configuration:
    model: str = 'convnext_base.fb_in22k_ft_in1k_384'
    img_size: int = 384
    batch_size: int = 6
    verbose: bool = True
    gpu_ids: tuple = (1,)
    normalize_features: bool = True
    checkpoint_start: str = './convnext_base.fb_in22k_ft_in1k_384/weights_e40_0.7786.pth'
    num_workers: int = 0 if os.name == 'nt' else 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ground_cutting: int = 0

def image_model(config: Configuration):
    model = TimmModel(config.model, pretrained=True, img_size=config.img_size)
    if config.checkpoint_start is not None:
        print("Start from:", config.checkpoint_start)
        model_state_dict = torch.load(config.checkpoint_start)
        model.load_state_dict(model_state_dict, strict=False)
    return model
  
class ImageEncoder(nn.Module):
    def __init__(self, config: Configuration, in_dim=1024):
        super().__init__()
        self.image_encoder = image_model(config)
        self.in_dim = in_dim

    def forward(self, image):
        feature_map = self.image_encoder(image)
        # Flatten the spatial dimensions and reshape
        batch_size, num_channels, height, width = feature_map.shape
        num_patches = height * width
        feature_map = feature_map.permute(0, 2, 3, 1).reshape(batch_size, num_patches, num_channels)
        # Ensure the feature dimension matches in_dim
        assert num_channels == self.in_dim, "Feature dimension mismatch"
        return feature_map

def text_encoder(input_text, model='text-embedding-3-small'):
    client = OpenAI(api_key='GPT_API_KEY')
    text_embedding = client.embeddings.create(input=[input_text], model=model).data[0].embedding
    return torch.tensor(text_embedding)

class ReRankingModule(nn.Module):
    def __init__(self, image_dim, text_dim, common_dim, dropout_rate=0.3):
        super().__init__()
        self.image_projection = nn.Linear(image_dim, common_dim)
        self.text_projection = nn.Linear(text_dim, common_dim)
        self.norm = nn.LayerNorm(common_dim)

        self.dense1 = nn.Linear(common_dim, 256)
        self.norm1 = nn.LayerNorm(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.dense2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.dense3 = nn.Linear(128, 64)
        self.norm3 = nn.LayerNorm(64)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(64, 1)

    def forward(self, query_img, query_text, ref_img, ref_text):
        query = self.image_projection(query_img) + self.text_projection(query_text)
        ref = self.image_projection(ref_img) + self.text_projection(ref_text)
        combined = query + ref
        combined = self.norm(combined)

        x = F.relu(self.norm1(self.dense1(combined)))
        x = self.dropout1(x)

        x = F.relu(self.norm2(self.dense2(x)))
        x = self.dropout2(x)

        x = F.relu(self.norm3(self.dense3(x)))
        x = self.dropout3(x)

        score = self.output(x).squeeze(-1)
        
        score = torch.sigmoid(score)
        return score