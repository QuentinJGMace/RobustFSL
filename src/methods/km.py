"""
Abstract class that is used for paddle type methods
"""

import torch
import torch.nn.functional as F
from src.methods.utils import get_one_hot
from src.methods.abstract_method import AbstractMethod, MinMaxScaler


class KM(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lambd = args.lambd
        self.init_info_lists()

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        n_tasks = samples.size(0)
        logits = (
            samples.matmul(self.w.transpose(1, 2))
            - 1 / 2 * (self.w**2).sum(2).view(n_tasks, 1, -1)
            - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1)
        )  #
        return logits

    def predict(self):
        """
        returns:
            preds : torch.Tensor of shape [n_task, n_query]
        """
        preds = self.u.argmax(2)
        return preds

    def init_w(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s, self.n_class)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.w = weights / counts

    def run_task(self, task_dic, shot):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic["y_s"]  # [n_task, shot]
        y_q = task_dic["y_q"]  # [n_task, n_query]
        support = task_dic["x_s"]  # [n_task, shot, feature_dim]
        query = task_dic["x_q"]  # [n_task, n_query, feature_dim]
        x_mean = task_dic["x_mean"]  # [n_task, feature_dim]
        idx_outliers_support = task_dic["outliers_support"].to(self.device)
        idx_outliers_query = task_dic["outliers_query"].to(self.device)

        # Transfer tensors to GPU if needed
        support = support.to(self.device)
        query = query.to(self.device)
        x_mean = x_mean.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations
        support, query = self.normalizer(support, query, train_mean=x_mean)

        # Run adaptation
        self.run_method(
            support=support,
            query=query,
            y_s=y_s,
            y_q=y_q,
            idx_outliers_support=idx_outliers_support,
            idx_outliers_query=idx_outliers_query,
        )

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_method(
        self, support, query, y_s, y_q, idx_outliers_support, idx_outliers_query
    ):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
            idx_outliers_support : torch.Tensor of shape [n_task, n_outliers_support]
            idx_outliers_query : torch.Tensor of shape [n_task, n_outliers_query]
        """
        pass

    def get_criterions(self, w_old, u_old):
        """
        Returns the criterions
        inputs:
            w_old : torch.Tensor of shape [n_task, num_class, feature_dim]
            u_old : torch.Tensor of shape [n_task, n_query, n_class]
        outputs:
            criterions : torch.Tensor of shape [n_task]
        """
        with torch.no_grad():
            crit_prot = torch.norm(self.w - w_old, dim=(-1, -2)).mean().item()
            crit_u = torch.norm(self.u - u_old, dim=(-1, -2)).mean().item()
        return {
            "crit_prot": crit_prot,
            "crit_u": crit_u,
        }
