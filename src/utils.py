
from pathlib import Path
import math
import numpy as np
from scipy.spatial import KDTree

def results_dir(cfg):
    folder = [f.name for f in Path(cfg.system.results_path).iterdir() if f.is_dir()]
    folder = [int(f) for f in folder if f.isdigit()]
    folder = max(folder) + 1 if folder else 0
    results_folder = f'{cfg.system.results_path}{folder}'
    Path(results_folder).mkdir(parents=True, exist_ok=True)
    Path(f'{results_folder}/ckpts').mkdir(parents=True, exist_ok=True)
    Path(f'{results_folder}/lightning_logs').mkdir(parents=True, exist_ok=True)
    return results_folder

def recall_accuracy(query, db):
    db_length = len(db)

    tree = KDTree(db)
    ks = [1, 5, 10]    
    metrics = {k: 0 for k in ks}

    _, retrievals = tree.query(query, k=db_length)

    for gt_ind, ret_inds in enumerate(retrievals):
        # Single Image Retrieval Recall Accuracies
        for k in filter(lambda k: len(np.intersect1d(ret_inds[:k], gt_ind)) > 0, ks):
            metrics[k] += 1

    for m in metrics:
        metrics[m] = round((metrics[m]/len(query))*100, 4)

    return metrics