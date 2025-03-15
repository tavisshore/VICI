
import math
import numpy as np
from scipy.spatial import KDTree


def recall_accuracy(emb_a, emb_b):
    db = np.concatenate(emb_a, axis=0)
    query = np.concatenate(emb_b, axis=0)
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
        metrics[m] = round((metrics[m]/db_length)*100, 4)

    return metrics