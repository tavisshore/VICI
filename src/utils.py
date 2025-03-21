
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
import timm 


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
    db_length = len(db)

    tree = KDTree(db)
    ks = [1, 5, 10]    
    metrics = {k: 0 for k in ks}

    _, retrievals = tree.query(query, k=db_length)

    for gt_ind, ret_inds in enumerate(retrievals):

        ground_truth_inds = np.where(labels == labels[gt_ind])[0]

        # Single Image Retrieval Recall Accuracies
        for k in filter(lambda k: len(np.intersect1d(ret_inds[:k], ground_truth_inds)) > 0, ks):
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
                224: 'timm/convnextv2_tiny.fcmae_ft_in22k_in1k_224',
                384: 'timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384'
            },
            'base': {
                224: 'timm/convnextv2_base.fcmae_ft_in22k_in1k_224',
                384: 'timm/convnextv2_base.fcmae_ft_in22k_in1k_384'
            }
        },  
        'dinov2': {
            'tiny': 'timm/vit_small_patch14_reg4_dinov2.lvd142m', # size irrelevant?
            'base': 'timm/vit_base_patch14_reg4_dinov2.lvd142m'
        },
        'vit': {
            'tiny': {
                224: 'timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k',
            },
            'base': {
                224: 'timm/vit_base_patch16_224.augreg_in21k_ft_in1k',
                384: 'timm/vit_base_patch16_384.augreg_in21k_ft_in1k'
            }   
        }
    }

    assert cfg.model.backbone in backbones, f"Backbone {cfg.model.backbone} not supported"
    assert cfg.model.size in backbones[cfg.model.backbone], f"Size {cfg.model.size} not supported for {cfg.model.backbone}"
    assert cfg.model.image_size in backbones[cfg.model.backbone][cfg.model.size], f"Image size {cfg.model.image_size} not supported for {cfg.model.backbone} {cfg.model.size}"

    network = backbones[cfg.model.backbone][cfg.model.size]
    if cfg.model.backbone != 'dinov2':
        network = network[cfg.model.image_size]

    return timm.create_model(backbones[cfg.model.backbone][cfg.model.size][cfg.model.image_size], pretrained=True, num_classes=0)

    
