import torch.nn.functional as F
from tqdm import tqdm
import torch
import time
from numpy import linalg as LA
import numpy as np
from scipy.stats import mode
from src.methods.abstract_method import AbstractMethod
from src.methods.utils import get_one_hot, get_metric


class BDCSPN(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone, device, log_file, args)
        self.temp = args.temp
        self.num_NN = args.num_NN
        self.number_tasks = self.args.batch_size
        self.device = "cpu"  # actually can't be on cuda
        self.init_info_lists()

    def record_acc(self, y_q, preds_q, idx_outliers_query):
        """
        inputs:
            y_q : torch.Tensor of shape [n_tasks, n_query]
            q_pred : torch.Tensor of shape [n_tasks, n_query]:
            idx_outliers_query : torch.Tensor of shape [n_tasks, n_outliers_query]
        """
        preds_q = torch.from_numpy(preds_q)
        y_q = torch.from_numpy(y_q)
        total_accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(total_accuracy)

        if (idx_outliers_query is not None) and (idx_outliers_query.shape[1] != 0):
            mask_outliers = torch.ones_like(y_q).bool().to(y_q.device)
            mask_outliers.scatter_(1, idx_outliers_query, False)

            iid_accuracy = (preds_q[mask_outliers] == y_q[mask_outliers]).float()
            iid_accuracy = iid_accuracy.mean(0).unsqueeze(0).expand((y_q.size(0), 1))
            self.iid_test_acc.append(iid_accuracy)
        else:
            self.iid_test_acc.append(total_accuracy)

    def proto_rectification(self, y_s, support, query, shot):
        """
        inputs:
            support : np.Array of shape [n_task, shot, feature_dim]
            query : np.Array of shape [n_task, n_query, feature_dim]
            shot: scalar

        ouput:
            proto_weights: prototype of each class [n_task, n_query, num_classes]
        """
        eta = support.mean(1) - query.mean(1)  # Shifting term
        query = (
            query + eta[:, np.newaxis, :]
        )  # Adding shifting term to each normalized query feature

        query_aug = torch.cat((support, query), axis=1)  # Augmented set S' (X')
        one_hot = get_one_hot(y_s, self.n_class)
        counts = one_hot.sum(1).view(support.size()[0], -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        init_prototypes = weights / counts

        proto_weights = []
        for j in range(self.number_tasks):

            distance = get_metric("cosine")(init_prototypes[j], query_aug[j])
            predict = torch.argmin(distance, dim=1)
            cos_sim = F.cosine_similarity(
                query_aug[j][:, None, :], init_prototypes[j][None, :, :], dim=2
            )  # Cosine similarity between X' and Pn
            cos_sim = self.temp * cos_sim
            W = F.softmax(cos_sim, dim=1)
            init_prototypeslist = [
                (W[predict == i, i].unsqueeze(1) * query_aug[j][predict == i]).mean(
                    0, keepdim=True
                )
                for i in predict.unique()
            ]
            proto = torch.cat(init_prototypeslist, dim=0)  # Rectified prototypes P'n

            if proto.shape[0] != len(torch.unique(y_s)):
                proto = init_prototypes[j]

            proto_weights.append(proto)

        proto_weights = np.stack(proto_weights, axis=0)
        return proto_weights

    def run_task(self, task_dic, shot):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic["y_s"]  # [n_task, shot]
        y_q = task_dic["y_q"]  # [n_task, n_query]
        x_s = task_dic["x_s"]  # [n_task, shot, feature_dim]
        x_q = task_dic["x_q"]  # [n_task, n_query, feature_dim]
        x_mean = task_dic["x_mean"]

        idx_outliers_support = task_dic["outliers_support"].to(self.device)
        idx_outliers_query = task_dic["outliers_query"].to(self.device)

        # Extract features
        support = x_s.to(self.device)
        query = x_q.to(self.device)
        support, query = self.normalizer(x_s, x_q, train_mean=x_mean)
        y_s = y_s.long().squeeze(2)
        y_q = y_q.long().squeeze(2)
        query = query.to("cpu")
        self.logger.info(" ==> Executing proto-rectification ...")
        support = self.proto_rectification(
            y_s=y_s, support=support, query=query, shot=shot
        )
        query = query.numpy()
        y_q = y_q.numpy()

        # Run method
        self.run_method(
            support=support,
            query=query,
            y_s=y_s,
            y_q=y_q,
            idx_outliers_support=idx_outliers_query,
            idx_outliers_query=idx_outliers_query,
            shot=shot,
        )

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_method(
        self, support, query, y_s, y_q, idx_outliers_support, idx_outliers_query, shot
    ):
        """
        Corresponds to the BD-CSPN inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
            idx_outliers_support : torch.Tensor of shape [n_task, n_outliers_support]
            idx_outliers_query : torch.Tensor of shape [n_task, n_outliers_query]
        """
        self.logger.info(" ==> Executing predictions on {} shot tasks ...".format(shot))
        out_list = []
        for i in range(self.number_tasks):
            t0 = time.time()
            y_s_i = np.unique(y_s[i])
            substract = support[i][:, None, :] - query[i]
            distance = LA.norm(substract, 2, axis=-1)
            idx = np.argpartition(distance, self.num_NN, axis=0)[: self.num_NN]
            nearest_samples = np.take(y_s_i, idx)
            out = mode(nearest_samples, axis=0)[0]
            t1 = time.time()
            self.record_convergence(timestamp=t1 - t0, criterions={})
            out_list.append(out)

        n_tasks, n_query, feature_dim = query.shape
        out = np.stack(out_list, axis=0).reshape((n_tasks, n_query))
        self.record_acc(y_q=y_q, preds_q=out, idx_outliers_query=idx_outliers_query)
