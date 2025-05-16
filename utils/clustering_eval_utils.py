from unittest import case
from tqdm import tqdm

import numpy as np
import sklearn
from sklearn.metrics import silhouette_score

# this function is probably useless as it's only a wrapper for the sklearn one now
# TODO: remove it if it doesn't change
def compute_silhouette_score(data, cluster_pred, metric='euclidean'):
    return silhouette_score(data, cluster_pred, metric=metric)

def evaluate_model(model, data, num_evals, aggregate='mean'):
    """
    compute silhouette_score multiple times to check for different fits
    :param model: initialized sklearn clustering model
    :param data: data to fit
    :param num_evals: number of clustering rounds
    :param aggregate: float or list [num_evals], how the scores are aggregated, defaults to 'mean', None leads to no aggregation
    :return:
    """
    scores = list()
    for i in tqdm(range(num_evals)):
        model = sklearn.base.clone(model)
        fit_res = model.fit(data)
        score = silhouette_score(data, fit_res.labels_)
        scores.append(score)
    match aggregate:
        case 'mean':  return np.mean(scores)
        case 'max': return np.max(scores)
        case 'min': return np.min(scores)
        case 'median': return np.median(scores)
        case None: return scores
