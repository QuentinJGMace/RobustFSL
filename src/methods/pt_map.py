# Adaptation of the publicly available code of the paper entitled "Open-Set Likelihood Maximization for Few-Shot Learning":
# https://github.com/ebennequin/few-shot-open-set

# Original method from the paper: Leveraging the Feature Distribution in Transfer-based Few-Shot Learning
# https://github.com/yhu01/PT-MAP

import torch
import time
from src.methods.abstract_method import AbstractMethod
from src.methods.utils import get_one_hot
from src.dataset import NORMALIZERS


class PT_MAP(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super(PT_MAP, self).__init__(backbone, device, log_file, args)
        self.alpha = args.alpha_pt
        self.lam = args.lam
        if self.args.normalizer == "default":
            self.normalizer = NORMALIZERS["transductive"]
        self.init_info_lists()

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

        indexes_outliers_support = task_dic["outliers_support"].to(self.device)
        indexes_outliers_query = task_dic["outliers_query"].to(self.device)

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
            support_features=support,
            query_features=query,
            y_s=y_s,
            y_q=y_q,
            indexes_outliers_support=indexes_outliers_support,
            indexes_outliers_query=indexes_outliers_query,
        )

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_method(
        self,
        support_features,
        query_features,
        y_s,
        y_q,
        indexes_outliers_support,
        indexes_outliers_query,
    ):
        """
        Args :
            support_features : torch.Tensor of shape [n_task, n_support, feature_dim]
            query_features : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, n_support]
            y_q : torch.Tensor of shape [n_task, n_query]
            indexes_outliers_support : torch.Tensor of shape [n_task, n_outliers_support]
            indexes_outliers_query : torch.Tensor of shape [n_task, n_outliers_query]
        """
        support_labels, query_labels = y_s, y_q

        self.prototypes = self.compute_prototypes(support_features, support_labels)
        probs_s = get_one_hot(support_labels, self.n_class)
        all_features = torch.cat([support_features, query_features], 1)
        for epoch in range(self.n_iter):
            t0 = time.time()
            probs_q = self.get_probas(query_features)
            all_probs = torch.cat([probs_s, probs_q], dim=1)

            # update centroids
            self.update_prototypes(all_features, all_probs)

            self.record_convergence(time.time() - t0, {})

        # get final accuracy and return it
        probs_s = self.get_probas(support_features).cpu()
        probs_q = self.get_probas(query_features).cpu()

        self.record_acc(
            probs_q, query_labels, indexes_outliers_query=indexes_outliers_query
        )

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        """
        M: [batch_size, N, K]
        r: [batch_size, N]
        c: [batch_size, K]
        """

        r = r.cuda()
        c = c.cuda()
        b, n, m = M.shape
        P = torch.exp(-self.lam * M)
        P /= P.sum(dim=(1, 2), keepdim=True)

        u = torch.zeros(b, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u + 1e-10).view(b, -1, 1)
            P *= (c / P.sum(1) + 1e-10).view(b, 1, -1)
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M, dim=(1, 2))

    def get_probas(self, query_features):
        """
        query_features: [batch_size, Nq, d]
        """

        dist = torch.cdist(query_features, self.prototypes) ** 2  # [batch_size, Nq, K]

        n_tasks = dist.size(0)
        n_usamples = query_features.size(1)
        n_ways = dist.size(2)

        r = torch.ones(n_tasks, n_usamples)
        c = torch.ones(n_tasks, n_ways) * (n_usamples // n_ways)

        probas_q, _ = self.compute_optimal_transport(dist, r, c, epsilon=1e-6)
        return probas_q

    def compute_prototypes(self, support, y_s):
        """
        support: [batch_size, N, d]
        y_s: [batch_size, N]
        """
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s, self.n_class)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        return weights / counts

    def update_prototypes(self, features, probas):
        """
        features: [batch_size, N, d]
        probas: [batch_size, N, K]

        mus : [batch_size, K, d]
        """
        new_prototypes = (probas.transpose(1, 2) @ features) / probas.sum(1).unsqueeze(
            2
        )
        delta = new_prototypes - self.prototypes
        self.prototypes += self.alpha * delta

    def record_acc(self, probs_q, y_q, indexes_outliers_query=None):
        """
        Records the accuracy for each task

        Args:
            probs: torch.Tensor of shape [n_task, n_query, num_classes]
            y_q: torch.Tensor of shape [n_task, n_query]
        """
        preds_q = probs_q.argmax(2)
        total_accuracy = (preds_q.cpu() == y_q.cpu()).float().mean(1, keepdim=True)

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
