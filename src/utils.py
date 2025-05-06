
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
import timm 
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from torchmetrics.utilities import dim_zero_cat
import torch
from torch import Tensor
from torchmetrics import Metric

# Pure torch implementation of eval metrics to support DDP.
def compute_mAP(index, good_index, junk_index):
    ap = 0.0
    cmc = torch.zeros(len(index), dtype=torch.int)

    if good_index.numel() == 0:
        cmc[0] = -1
        return ap, cmc

    # Remove junk indices
    mask = ~torch.isin(index, junk_index)
    index = index[mask]

    # Find good index positions
    ngood = good_index.numel()
    mask = torch.isin(index, good_index)
    rows_good = torch.nonzero(mask).flatten()

    if rows_good.numel() == 0:
        cmc[0] = -1
        return ap, cmc

    cmc[rows_good[0]:] = 1  # CMC calculation

    for i in range(ngood):
        d_recall = 1.0 / ngood
        rank = rows_good[i].item()
        precision = (i + 1) / (rank + 1)
        old_precision = i / rank if rank != 0 else 1.0
        ap += d_recall * (old_precision + precision) / 2

    return ap, cmc

class CMCmAPMetric(Metric):
    def __init__(self, ranks=[1, 5, 10], **kwargs):
        super().__init__(**kwargs)
        self.ranks = ranks
        self.add_state("query_features", default=[], dist_reduce_fx="cat")
        self.add_state("query_ids", default=[], dist_reduce_fx="cat")
        self.add_state("gallery_features", default=[], dist_reduce_fx="cat")
        self.add_state("gallery_ids", default=[], dist_reduce_fx="cat")
    
    def update(self, feature: Tensor, id: int, branch: str):
        if branch == "satellite":
            self.gallery_features.append(feature)
            self.gallery_ids.append(torch.tensor([id], device=feature.device))
        elif branch == "streetview":
            self.query_features.append(feature)
            self.query_ids.append(torch.tensor([id], device=feature.device))
        else:
            raise NotImplementedError(f'{branch} is not implemented')
    
    def compute(self):
        query_features = dim_zero_cat(self.query_features)  # [Nq, D]
        query_ids = dim_zero_cat(self.query_ids)  # [Nq]
        gallery_features = dim_zero_cat(self.gallery_features)  # [Ng, D]
        gallery_ids = dim_zero_cat(self.gallery_ids)  # [Ng]

        cmc = torch.zeros(len(gallery_ids), dtype=torch.float, device=query_features.device)
        ap = 0.0
        count = 0

        for i in range(len(query_ids)):
            score = torch.matmul(gallery_features, query_features[i].unsqueeze(-1)).squeeze(-1)
            index = torch.argsort(score, descending=True)

            good_index = torch.nonzero(gallery_ids == query_ids[i]).flatten()
            junk_index = torch.nonzero(gallery_ids == -1).flatten()

            ap_i, cmc_i = compute_mAP(index, good_index, junk_index)

            if cmc_i[0] == -1:
                continue

            cmc += cmc_i.float().to(cmc.device)
            ap += ap_i
            count += 1

        cmc /= count
        ap = (ap / count) * 100

        string = []
        for i in self.ranks:
            string.append('Recall@{}: {:.4f}'.format(i, cmc[i - 1] * 100))
        string.append('AP: {:.4f}'.format(ap))

        # print(' - '.join(results))
        return cmc


@rank_zero_only
def results_dir(cfg):
    folder = [f.name for f in Path(cfg.system.results_path).iterdir() if f.is_dir()]
    folder = [int(f) for f in folder if f.isdigit()]
    folder = max(folder) + 1 if folder else 0
    results_folder = f'{cfg.system.results_path}{folder}'
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    Path(f'{results_folder}/ckpts').mkdir(parents=True, exist_ok=True)
    Path(f'{results_folder}/lightning_logs').mkdir(parents=True, exist_ok=True)
    return results_folder

def recall_accuracy(query, db, labels):
    tree = KDTree(db)
    ks = [1, 5, 10]    
    metrics = {k: 0 for k in ks}

    _, retrievals = tree.query(query, k=10)

    # NOTE: How the data is setup currently - only one streetview to one satellite per epoch, randomly selected.
    ground_truths = {}
    for i, label in enumerate(labels):
        if label not in ground_truths:
            ground_truths[label] = []
        ground_truths[label].append(i)

    for gt_ind, ret_inds in enumerate(retrievals):
        indices = ground_truths[labels[gt_ind]]
        # print(f'{labels[gt_ind]}: {indices} - {ret_inds}')
        for k in filter(lambda k: len(np.intersect1d(ret_inds[:k], indices)) > 0, ks):
            metrics[k] += 1

    for m in metrics:
        metrics[m] = round((metrics[m]/len(query))*100, 4)

    return metrics


def get_backbone(cfg):
    # Better way to do this?
    # Add more backbones here to eval
    backbones = {
        'convnext': {
            'tiny': {
                224: 'timm/convnextv2_tiny.fcmae_ft_in22k_in1k',
                384: 'timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384'
            },
            'base': {
                224: 'timm/convnextv2_base.fcmae_ft_in22k_in1k',
                384: 'timm/convnextv2_base.fcmae_ft_in22k_in1k_384'
            }
        },  
        'vit': {
            'tiny': {
                224: 'timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k',
            },
            'small': {
                224: 'timm/vit_small_patch16_224.augreg_in21k_ft_in1k',
                384: 'timm/vit_small_patch16_384.augreg_in21k_ft_in1k'
            },
            'base': {
                224: 'timm/vit_base_patch16_224.augreg_in21k_ft_in1k',
                384: 'timm/vit_base_patch16_384.augreg_in21k_ft_in1k'
            }   
        },
        'dinov2': {
            'tiny': {
                518: 'timm/vit_small_patch14_reg4_dinov2.lvd142m',
            },
            'base': {
                518: 'timm/vit_base_patch14_reg4_dinov2.lvd142m',
             },
             'large': {
                518: 'timm/vit_large_patch14_dinov2.lvd142m',
             },
        },
        'eva02':{
            'base':{
                224: 'timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k',
            },
            'large':{
                448: 'timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k',
            },
            'enormous':{
                224: 'timm/eva02_enormous_patch14_plus_clip_224.laion2b_s9b_b144k'
            },
        }
    }

    assert cfg.model.backbone in backbones, f"Backbone {cfg.model.backbone} not supported"
    assert cfg.model.size in backbones[cfg.model.backbone], f"Size {cfg.model.size} not supported for {cfg.model.backbone}"
    assert cfg.model.image_size in backbones[cfg.model.backbone][cfg.model.size], f"Image size {cfg.model.image_size} not supported for {cfg.model.backbone} {cfg.model.size}"

    model = timm.create_model(backbones[cfg.model.backbone][cfg.model.size][cfg.model.image_size], pretrained=True, num_classes=0)

    # ensure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True
    # set the model to train mode
    model.train()

    return model

    
if __name__ == '__main__':
    model = timm.create_model('timm/eva02_base_patch16_clip_224.merged2b_s8b_b131k', pretrained=True, num_classes=0)
