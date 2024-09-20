from collections import defaultdict

import torch.nn.functional as F
from tqdm import tqdm
import math
import torch
import time
import numpy as np
from numpy import linalg as LA
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from src.methods.abstract_method import AbstractMethod
from src.methods.utils import get_one_hot, get_metric


class LaplacianShot(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone, device, log_file, args)
        self.knn = args.knn
        self.arch = args.arch
        self.proto_rect = args.proto_rect
        self.norm_type = args.norm_type
        self.lmd = args.lmd
        self.temp = args.temp
        self.init_info_lists()

    def __del__(self):
        self.logger.del_logger()

    def init_info_lists(self):
        """
        Initializes the lists for logging
        """
        self.timestamps = []
        self.ent_energy = []
        self.criterions = defaultdict(lambda: [])
        self.test_acc = []

    def record_acc(self, acc, ent_energy):
        """
        inputs:
            acc_list : torch.Tensor of shape [iter]
            ent_energy : torch.Tensor of shape [iter]
            new_time: torch.Tensor of shape [iter]
        """

        self.test_acc.append(acc)
        self.ent_energy.append(ent_energy)

    def get_logs(self):
        """
        Returns the logs
        outputs:
            logs : dict {"timestamps": list, "criterions": np.array, "acc": np.array}
        """
        for key in self.criterions.keys():
            self.criterions[key] = np.array(self.criterions[key])
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        self.ent_energy = np.array(self.ent_energy)
        return {
            "timestamps": self.timestamps,
            "criterions": self.criterions,
            "acc": self.test_acc,
            "ent_energy": self.ent_energy,
        }

    def proto_rectification(self, y_s, support, query, shot):
        """
        inputs:
            support : np.Array of shape [n_task, s_shot, feature_dim]
            query : np.Array of shape [n_task, q_shot, feature_dim]
            shot: Shot

        ouput:
            proto_weights: prototype of each class
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

    def create_affinity(self, X):
        N, D = X.shape

        nbrs = NearestNeighbors(n_neighbors=self.knn).fit(X)
        dist, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), self.knn - 1)
        col = knnind[:, 1:].flatten()
        data = np.ones(X.shape[0] * (self.knn - 1))
        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
        return W

    def normalize(self, Y_in):
        maxcol = np.max(Y_in, axis=1)
        Y_in = Y_in - maxcol[:, np.newaxis]
        N = Y_in.shape[0]
        size_limit = 150000
        if N > size_limit:
            batch_size = 1280
            Y_out = []
            num_batch = int(math.ceil(1.0 * N / batch_size))
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, N)
                tmp = np.exp(Y_in[start:end, :])
                tmp = tmp / (np.sum(tmp, axis=1)[:, None])
                Y_out.append(tmp)
            del Y_in
            Y_out = np.vstack(Y_out)
        else:
            Y_out = np.exp(Y_in)
            Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

        return Y_out

    def entropy_energy(self, Y, unary, kernel, bound_lambda, batch=False):
        tot_size = Y.shape[0]
        pairwise = kernel.dot(Y)
        if batch == False:
            temp = (unary * Y) + (-bound_lambda * pairwise * Y)
            E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
        else:
            batch_size = 1024
            num_batch = int(math.ceil(1.0 * tot_size / batch_size))
            E = 0
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, tot_size)
                temp = (unary[start:end] * Y[start:end]) + (
                    -bound_lambda * pairwise[start:end] * Y[start:end]
                )
                E = (
                    E
                    + (
                        Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp
                    ).sum()
                )

        return E

    def bound_update(
        self,
        unary,
        kernel,
        bound_lambda,
        y_s,
        y_q,
        task_i,
        bound_iteration=20,
        batch=False,
    ):
        oldE = float("inf")
        Y = self.normalize(-unary)
        E_list = []
        out_list = []
        acc_list = []
        timestamps = []
        t0 = time.time()
        for i in range(bound_iteration):
            additive = -unary
            mul_kernel = kernel.dot(Y)
            Y = -bound_lambda * mul_kernel
            additive = additive - Y
            Y = self.normalize(additive)
            E = self.entropy_energy(Y, unary, kernel, bound_lambda, batch)
            E_list.append(E)
            # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
            l = np.argmax(Y, axis=1)
            # out = np.take(y_s, l)
            out = l
            timestamps.append(time.time() - t0)

            if i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE)):
                # print('Converged')
                out_list.append(torch.from_numpy(out))
                acc_list.append(
                    (torch.from_numpy(y_q[task_i]) == torch.from_numpy(out)).float()
                )
                for j in range(bound_iteration - i - 1):
                    out_list.append(out_list[i].detach().clone())
                    acc_list.append(acc_list[i].detach().clone())
                    E_list.append(E_list[i])
                    timestamps.append(0)
                break

            else:
                oldE = E.copy()

                out_list.append(torch.from_numpy(out))
                acc_list.append(
                    (torch.from_numpy(y_q[task_i]) == torch.from_numpy(out)).float()
                )
            t0 = time.time()

        out_list = torch.stack(out_list, dim=0)
        acc_list = torch.stack(acc_list, dim=0).mean(dim=1, keepdim=True)

        return out, acc_list, E_list, timestamps

    def run_task(self, task_dic, shot):
        # Extract support and query
        y_s, y_q = task_dic["y_s"], task_dic["y_q"]
        x_s, x_q = task_dic["x_s"], task_dic["x_q"]
        train_mean = task_dic["x_mean"]

        # Extract features
        y_s = y_s.squeeze(2).to(self.device)
        y_q = y_q.squeeze(2).to(self.device)
        x_s = x_s.to(self.device)
        x_q = x_q.to(self.device)
        support, query = self.normalizer(support=x_s, query=x_q, train_mean=train_mean)

        if self.proto_rect:
            self.logger.info(" ==> Executing proto-rectification ...")
            support = self.proto_rectification(
                y_s=y_s, support=support, query=query, shot=shot
            )
        else:
            one_hot = get_one_hot(y_s, self.n_class)
            counts = one_hot.sum(1).view(support.size()[0], -1, 1)
            weights = one_hot.transpose(1, 2).matmul(support)
            support = weights / counts
            support = support.cpu().numpy()
        query = query.cpu().numpy()
        y_s = y_s.cpu().numpy()
        y_q = y_q.cpu().numpy()

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def run_method(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the LaplacianShot inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
        records :
            accuracy
            ent_energy
            inference time
        """
        self.logger.info(
            " ==> Executing {}-shot predictions with lmd = {} ...".format(
                shot, self.lmd
            )
        )
        n_tasks = support.shape[0]
        for i in range(n_tasks):

            t0 = time.time()
            substract = support[i][:, None, :] - query[i]
            distance = LA.norm(substract, 2, axis=-1)
            unary = distance.transpose() ** 2
            W = self.create_affinity(query[i])
            preds, acc_list, ent_energy, times = self.bound_update(
                unary=unary,
                kernel=W,
                bound_lambda=self.lmd,
                y_s=y_s,
                y_q=y_q,
                task_i=i,
                bound_iteration=self.n_iter,
            )
            t1 = time.time()
            self.record_acc(acc=acc_list, ent_energy=ent_energy)
            self.record_convergence(timestamp=t1 - t0, criterions={})
