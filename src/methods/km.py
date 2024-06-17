"""
Abstract class that is used for paddle type methods
"""

import torch
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

    def record_acc(self, y_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """
        preds_q = self.predict()
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def get_logs(self):

        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {
            "timestamps": self.timestamps,
            "criterions": self.criterions,
            "acc": self.test_acc,
        }

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

        # Transfer tensors to GPU if needed
        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations
        scaler = MinMaxScaler(feature_range=(0, 1))
        query, support = scaler(query, support)

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs
