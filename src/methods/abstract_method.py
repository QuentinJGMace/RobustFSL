from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from src.logger import Logger
import matplotlib.pyplot as plt
from src.dataset import NORMALIZERS


class AbstractMethod(nn.Module):
    """Abstract class for few shot learning methods"""

    def __init__(self, backbone, device, log_file, args):
        super(AbstractMethod, self).__init__()
        self.device = device
        self.n_iter = args.iter
        self.n_class = args.n_class_support
        self.backbone = backbone
        self.log_file = log_file
        self.logger = Logger(__name__, self.log_file)
        if args.normalizer != "default":
            self.normalizer = NORMALIZERS[args.normalizer]
        else:
            self.normalizer = NORMALIZERS[
                "paddle"
            ]  # used by most methods, may be overwritten if the normalizer is not the correct one for the method
        self.args = args

    def __del__(self):
        if hasattr(self, "logger"):
            self.logger.del_logger()

    def init_info_lists(self):
        """
        Initializes the lists for logging
        """
        self.timestamps = []
        self.criterions = defaultdict(lambda: [])
        self.test_acc = []
        self.iid_test_acc = []

    def record_convergence(self, timestamp, criterions):
        """
        Records the logs
        inputs:
            timestamp : float
            criterion : torch.Tensor of shape [n_task]
        """
        self.timestamps.append(timestamp)
        for key, value in criterions.items():
            self.criterions[key].append(value)

    def plot_convergence(self, dirpath):
        """
        Plots the convergence plot
        inputs:
            dirpath : str
        """
        raise NotImplementedError("Eh t'as oublié de me faire")
        plt.plot(range(len(self.criterions)), self.criterions)
        plt.xlabel("Iter")
        plt.ylabel("Criterion")
        plt.savefig(filepath)
        plt.close()

    def record_acc(self, y_q, indexes_outliers_query=None):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
            indexes_outliers_query : torch.Tensor of shape [n_task, n_outliers_query] : indexes of the outliers in the query set
        """
        preds_q = self.predict()
        total_accuracy = (preds_q == y_q).float().mean(1, keepdim=True)

        self.test_acc.append(total_accuracy)

        if (indexes_outliers_query is not None) and (
            indexes_outliers_query.shape[1] != 0
        ):
            mask_outliers = torch.ones_like(y_q).bool().to(y_q.device)
            mask_outliers.scatter_(1, indexes_outliers_query, False)

            iid_accuracy = (preds_q[mask_outliers] == y_q[mask_outliers]).float()
            iid_accuracy = iid_accuracy.mean(0).unsqueeze(0).expand((y_q.size(0), 1))
            self.iid_test_acc.append(iid_accuracy)
        else:
            self.iid_test_acc.append(total_accuracy)

    def get_logs(self):
        """
        Returns the logs
        outputs:
            logs : dict {"timestamps": list, "criterions": np.array, "acc": np.array}
        """
        for key in self.criterions.keys():
            self.criterions[key] = np.array(self.criterions[key])
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        self.iid_test_acc = torch.cat(self.iid_test_acc, dim=1).cpu().numpy()
        return {
            "timestamps": self.timestamps,
            "criterions": self.criterions,
            "acc": self.test_acc,
            "iid_acc": self.iid_test_acc,
        }

    def push_logs(self, info, verbose=False):
        """
        Pushes the info to the log file
        inputs:
            info : str
            verbose : bool
        """
        with open(self.log_file, "a") as f:
            f.write(info + "\n")

        if verbose:
            print(info)
        return

    def get_logits(self, samples):
        """
        Returns the logits of the samples
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        outputs:
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        pass

    def predict(self):
        """
        Returns the predictions
        outputs:
            preds : torch.Tensor of shape [n_task, n_query]
        """
        pass

    def run_task(self, task_dic, shot):
        """
        Fits the model to the support set
        inputs:
            task_dic : dict {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int
        """
        pass


class MinMaxScaler(object):
    """MinMax Scaler

    Transforms each channel to the range [a, b].

    Parameters
    ----------
    feature_range : tuple
        Desired range of transformed data.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __call__(self, query, support):

        dist = query.max(dim=1, keepdim=True)[0] - query.min(dim=1, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        ratio = query.min(dim=1, keepdim=True)[0]
        query.mul_(scale).sub_(ratio)
        support.mul_(scale).sub_(ratio)
        return query, support
