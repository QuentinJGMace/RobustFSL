# Adaptation of the publicly available code of the NeurIPS 2020 paper entitled "TIM: Transductive Information Maximization":
# https://github.com/mboudiaf/TIM
from collections import defaultdict
import torch.nn.functional as F
from src.methods.abstract_method import AbstractMethod, MinMaxScaler
from src.api.utils import wrap_tqdm
from src.methods.utils import get_one_hot
from src.logger import Logger
from tqdm import tqdm
import torch
import time
import numpy as np
from copy import deepcopy


class RobustTIM(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.temp = args.temp
        self.loss_weights = args.loss_weights.copy()
        self.init_info_lists()
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support
        self.beta = args.beta
        self.id_cov = args.id_cov

    def init_info_lists(self):
        self.timestamps = []
        self.criterions = defaultdict(list)
        self.test_acc = []
        self.losses = []

    def compute_mahalanobis(self, samples, theta):
        """
        Computes the Mahalanobis distance between the samples and the prototypes
        inputs:
            samples : torch.Tensor of shape [n_task, n_sample, feature_dim]
            theta : torch.Tensor of shape [n_task, n_sample]
        """
        if not self.id_cov:
            tranformed_samples = torch.matmul(
                self.q.unsqueeze(1), samples.unsqueeze(2).unsqueeze(-1)
            ).squeeze(
                -1
            )  # Q_kx_n [n_task, n_sample, n_class, feature_dim]
        else:
            tranformed_samples = samples.unsqueeze(2)
        reshaped_theta = theta.unsqueeze(-1)  # theta shape : [n_task, n_sample, 1]
        # computes the norm of the difference between the samples and the prototypes
        dist_beta = (tranformed_samples - self.prototypes.unsqueeze(1)).norm(
            dim=-1
        ) ** self.beta  # ||Q_kx_n - prototype_k||^beta [n_task, n_sample, n_class]
        dist_beta /= reshaped_theta ** (
            self.beta - 1
        )  # ||Q_kx_n - prototype_k||^beta/theta [n_task, n_sample, n_class]

        return dist_beta

    def get_logits(self, samples, theta):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_query = samples.size(1)
        dist = self.compute_mahalanobis(samples, theta)
        logits = (-1 / 2) * self.temp * (dist)

        return logits

    def init_prototypes(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.backbone.eval()
        n_tasks, n_support, _ = support.size()
        _, n_query, _ = query.size()
        one_hot = get_one_hot(y_s, self.n_class).to(self.device)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.prototypes = weights / counts

        return

    def init_params(self, support, y_s, query):
        """
        Initializes the parameters
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
        """
        n_tasks, n_query, feature_dim = query.size()
        _, n_support, _ = support.size()

        self.init_prototypes(support, query, y_s)
        self.theta = torch.ones(n_tasks, n_support + n_query).to(self.device)
        self.u = (
            (self.get_logits(query, self.theta[:, -n_query:]))
            .softmax(-1)
            .to(self.device)
        )

        if not self.id_cov:
            # self.q strores the n_class covariances
            self.q = (
                torch.eye(feature_dim)
                .unsqueeze(0)
                .repeat(n_tasks, self.n_class, 1, 1)
                .to(self.device)
            )

    def record_acc(self, query, y_q):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_q : torch.Tensor of shape [n_task, n_query]
        """
        _, n_query, _ = query.size()
        logits_q = self.get_logits(query, self.theta[:, -n_query:]).detach()
        preds_q = logits_q.argmax(2)
        q_probs = logits_q.softmax(2)
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the baseline (no transductive inference = SimpleShot)
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        pass

    def compute_lambda(self, support, query):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]

        updates :
            self.loss_weights[0] : Scalar
        """
        self.N_s, self.N_q = support.size(1), query.size(1)
        # self.n_ways = torch.unique(y_s).size(0)
        if self.loss_weights[0] == "auto":
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

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

        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = x_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations required
        support = F.normalize(support, dim=2)
        query = F.normalize(query, dim=2)
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # query, support = scaler(query, support)
        support = support.to(self.device)
        query = query.to(self.device)

        # Initialize weights lambda
        self.compute_lambda(support=support, query=query)
        # Init basic prototypes
        self.init_params(support=support, y_s=y_s, query=query)

        # Run method
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def get_criterions(self, old_proto, old_theta):
        """
        inputs:
            old_prot : torch.Tensor of shape [n_task, num_class, feature_dim]

        returns:
            criterions : torch.Tensor of shape [n_task]
        """
        with torch.no_grad():
            crit_proto = (self.prototypes - old_proto).norm(dim=[1, 2]).mean().item()
            crit_theta = (self.theta - old_theta).norm(dim=-1).mean().item()

        return {
            "proto": crit_proto,
            "theta": crit_theta,
        }


class RTIM_GD(RobustTIM):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lr = float(args.lr_tim)
        self.optimizer = args.optimizer

    def init_params(self, support, y_s, query):
        super().init_params(support, y_s, query)
        self.theta = torch.nn.Parameter(self.theta)
        self.prototypes = torch.nn.Parameter(self.prototypes)
        if not self.id_cov:
            self.q = torch.nn.Parameter(self.q)

    def run_method(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()

        # self.prototypes.requires_grad_()
        if not self.id_cov:
            learnable_params = [self.prototypes, self.theta, self.q]
        else:
            learnable_params = [self.prototypes, self.theta]

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(learnable_params, lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(learnable_params, lr=self.lr)

        y_s_one_hot = get_one_hot(y_s, self.n_class)
        n_tasks, n_support, feature_dim = support.size()
        _, n_query, _ = query.size()

        for i in wrap_tqdm(range(self.n_iter), disable_tqdm=True):

            weights_old = deepcopy(self.prototypes.detach())
            theta_old = deepcopy(self.theta.detach())
            t0 = time.time()
            logits_s = self.get_logits(support, self.theta[:, :n_support])
            logits_q = self.get_logits(query, self.theta[:, -n_query:])

            ce = (
                -(y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12))
                .sum(2)
                .mean(1)
                .sum(0)
            )
            q_probs = logits_q.softmax(2)
            q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            q_ent = (
                -(q_probs.mean(1) * torch.log(q_probs.mean(1) + 1e-12)).sum(1).sum(0)
            )
            loss = self.loss_weights[0] * ce - (
                self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            criterions = self.get_criterions(weights_old, theta_old)
            self.record_convergence(timestamp=t1 - t0, criterions=criterions)

        self.backbone.eval()
        self.record_acc(query=query, y_q=y_q)


# class Alpha_TIM(TIM):
#     def __init__(self, backbone, device, log_file, args):
#         super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
#         self.lr = float(args.lr_alpha_tim)
#         self.entropies = args.entropies.copy()
#         self.alpha_values = args.alpha_value
#         # if alpha_values is not a list, convert it to list of 3 elements
#         if not isinstance(self.alpha_values, list):
#             self.alpha_values = [self.alpha_values] * 3

#     def run_method(self, support, query, y_s, y_q, shot):
#         """
#         Corresponds to the ALPHA-TIM inference
#         inputs:
#             support : torch.Tensor of shape [n_task, shot, feature_dim]
#             query : torch.Tensor of shape [n_task, n_query, feature_dim]
#             y_s : torch.Tensor of shape [n_task, shot]
#             y_q : torch.Tensor of shape [n_task, n_query]

#         updates :
#             self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
#         """

#         self.logger.info(
#             " ==> Executing ALPHA-TIM adaptation over {} iterations on {} shot tasks with alpha = {}...".format(
#                 self.n_iter, shot, self.alpha_values
#             )
#         )

#         self.prototypes.requires_grad_()
#         optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
#         y_s_one_hot = get_one_hot(y_s, self.n_class)
#         self.backbone.train()

#         for i in tqdm(range(self.n_iter)):
#             weights_old = deepcopy(self.prototypes.detach())
#             t0 = time.time()
#             logits_s = self.get_logits(support)
#             logits_q = self.get_logits(query)
#             q_probs = logits_q.softmax(2)

#             # Cross entropy type
#             if self.entropies[0] == "Shannon":
#                 ce = (
#                     -(y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12))
#                     .sum(2)
#                     .mean(1)
#                     .sum(0)
#                 )
#             elif self.entropies[0] == "Alpha":
#                 ce = torch.pow(y_s_one_hot, self.alpha_values[0]) * torch.pow(
#                     logits_s.softmax(2) + 1e-12, 1 - self.alpha_values[0]
#                 )
#                 ce = ((1 - ce.sum(2)) / (self.alpha_values[0] - 1)).mean(1).sum(0)
#             else:
#                 raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

#             # Marginal entropy type
#             if self.entropies[1] == "Shannon":
#                 q_ent = -(q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
#             elif self.entropies[1] == "Alpha":
#                 q_ent = (
#                     (1 - (torch.pow(q_probs.mean(1), self.alpha_values[1])).sum(1))
#                     / (self.alpha_values[1] - 1)
#                 ).sum(0)
#             else:
#                 raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

#             # Conditional entropy type
#             if self.entropies[2] == "Shannon":
#                 q_cond_ent = (
#                     -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
#                 )
#             elif self.entropies[2] == "Alpha":
#                 q_cond_ent = (
#                     (
#                         (1 - (torch.pow(q_probs + 1e-12, self.alpha_values[2])).sum(2))
#                         / (self.alpha_values[2] - 1)
#                     )
#                     .mean(1)
#                     .sum(0)
#                 )
#             else:
#                 raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

#             # Loss
#             loss = self.loss_weights[0] * ce - (
#                 self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent
#             )

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             t1 = time.time()
#             criterions = self.get_criterions(weights_old)
#             self.record_convergence(timestamp=t1 - t0, criterions=criterions)

#         self.backbone.eval()
#         self.record_acc(query=query, y_q=y_q)
