import numpy as np
import torch
import torch.nn.functional as F


def compute_confidence_interval(data, axis=0):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data. (assuming it is gaussian...)
    """
    a = 1.0 * np.array(data)
    m = np.mean(a, axis=axis)
    std = np.std(a, axis=axis)
    pm = 1.96 * (std / np.sqrt(a.shape[axis]))
    return m, pm


def get_metric(metric_type):
    METRICS = {
        "cosine": lambda gallery, query: 1.0
        - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        "euclidean": lambda gallery, query: (
            (query[:, None, :] - gallery[None, :, :]) ** 2
        ).sum(2),
        "l1": lambda gallery, query: torch.norm(
            (query[:, None, :] - gallery[None, :, :]), p=1, dim=2
        ),
        "l2": lambda gallery, query: torch.norm(
            (query[:, None, :] - gallery[None, :, :]), p=2, dim=2
        ),
    }
    return METRICS[metric_type]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
