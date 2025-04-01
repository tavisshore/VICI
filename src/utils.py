
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
import timm 
from lightning.pytorch.utilities.rank_zero import rank_zero_only

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
    db_length = len(db)

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
        for k in filter(lambda k: len(np.intersect1d(ret_inds[:k], indices)) > 0, ks):
            metrics[k] += 1

    for m in metrics:
        metrics[m] = round((metrics[m]/len(query))*100, 4)

    return metrics

# validate diagnal GT by using numpy
def validatenp(sat_global_descriptor, grd_global_descriptor):
    dist_array = 2.0 - 2.0 * np.matmul(sat_global_descriptor, grd_global_descriptor.T)
    
    top1_percent = 11
    val_accuracy = np.zeros((top1_percent))
    for i in range(top1_percent):
        # val_accuracy[0, i] = validate(dist_array, i)
        accuracy = 0.0
        data_amount = 0.0
        for k in range(dist_array.shape[0]):
            gt_dist = dist_array[k,k]
            prediction = np.sum(dist_array[:, k] < gt_dist)
            if prediction < i:
                accuracy += 1.0
            data_amount += 1.0
        accuracy /= data_amount
        val_accuracy[i] = accuracy * 100.0
    return val_accuracy



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
        'dinov2': {
            'tiny': 'timm/vit_small_patch14_reg4_dinov2.lvd142m', # size irrelevant?
            'base': 'timm/vit_base_patch14_reg4_dinov2.lvd142m'
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
        }
    }

    assert cfg.model.backbone in backbones, f"Backbone {cfg.model.backbone} not supported"
    assert cfg.model.size in backbones[cfg.model.backbone], f"Size {cfg.model.size} not supported for {cfg.model.backbone}"
    assert cfg.model.image_size in backbones[cfg.model.backbone][cfg.model.size], f"Image size {cfg.model.image_size} not supported for {cfg.model.backbone} {cfg.model.size}"

    network = backbones[cfg.model.backbone][cfg.model.size]
    if cfg.model.backbone != 'dinov2':
        network = network[cfg.model.image_size]

    return timm.create_model(backbones[cfg.model.backbone][cfg.model.size][cfg.model.image_size], pretrained=True, num_classes=0)

    
