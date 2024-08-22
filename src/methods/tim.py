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


class TIM(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.temp = args.temp
        self.loss_weights = args.loss_weights.copy()
        self.init_info_lists()
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support

    def init_info_lists(self):
        self.timestamps = []
        self.criterions = defaultdict(list)
        self.test_acc = []
        self.losses = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_tasks = samples.size(0)
        logits = self.temp * (
            samples.matmul(self.weights.transpose(1, 2))
            - 1 / 2 * (self.weights**2).sum(2).view(n_tasks, 1, -1)
            - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1)
        )  #
        return logits

    def init_weights(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.backbone.eval()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s, self.n_class).to(self.device)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts
        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)
        return q_probs

    def record_acc(self, query, y_q):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_q : torch.Tensor of shape [n_task, n_query]
        """
        logits_q = self.get_logits(query).detach()
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
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
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
        x_mean = task_dic["x_mean"]  # [feature_dim]

        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)  # [ N * (K_s + K_q), d]
        query = x_q.to(self.device)  # [ N * (K_s + K_q), d]
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations required
        # support = (support - x_mean.unsqueeze(1))
        # query = (query - x_mean.unsqueeze(1))
        # support = F.normalize(support, dim=2)
        # query = F.normalize(query, dim=2)
        # # scaler = MinMaxScaler(feature_range=(0, 1))
        # # query, support = scaler(query, support)
        support, query = self.normalizer(support, query, x_mean=x_mean)
        support = support.to(self.device)
        query = query.to(self.device)

        # Initialize weights lambda
        self.compute_lambda(support=support, query=query)
        # Init basic prototypes
        self.init_weights(support=support, y_s=y_s, query=query)

        # Run method
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q, shot=shot)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

    def get_criterions(self, old_proto):
        """
        inputs:
            old_prot : torch.Tensor of shape [n_task, num_class, feature_dim]

        returns:
            criterions : torch.Tensor of shape [n_task]
        """
        with torch.no_grad():
            crit_proto = (self.weights - old_proto).norm(dim=[1, 2]).mean().item()

        return {
            "proto": crit_proto,
        }


class TIM_GD(TIM):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lr = float(args.lr_tim)

    def run_method(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        t0 = time.time()

        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s, self.n_class)
        # self.backbone.train()

        for i in wrap_tqdm(range(self.n_iter), disable_tqdm=True):

            weights_old = deepcopy(self.weights.detach())
            t0 = time.time()
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)

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
            criterions = self.get_criterions(weights_old)
            self.record_convergence(timestamp=t1 - t0, criterions=criterions)

        self.backbone.eval()
        self.record_acc(query=query, y_q=y_q)


class Alpha_TIM(TIM):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lr = float(args.lr_alpha_tim)
        self.entropies = args.entropies.copy()
        self.alpha_values = args.alpha_value
        # if alpha_values is not a list, convert it to list of 3 elements
        if not isinstance(self.alpha_values, list):
            self.alpha_values = [self.alpha_values] * 3

    def run_method(self, support, query, y_s, y_q, shot):
        """
        Corresponds to the ALPHA-TIM inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        self.logger.info(
            " ==> Executing ALPHA-TIM adaptation over {} iterations on {} shot tasks with alpha = {}...".format(
                self.n_iter, shot, self.alpha_values
            )
        )

        self.weights.requires_grad_()
        optimizer = torch.optim.Adam([self.weights], lr=self.lr)
        y_s_one_hot = get_one_hot(y_s, self.n_class)
        self.backbone.train()

        for i in tqdm(range(self.n_iter)):
            weights_old = deepcopy(self.weights.detach())
            t0 = time.time()
            logits_s = self.get_logits(support)
            logits_q = self.get_logits(query)
            q_probs = logits_q.softmax(2)

            # Cross entropy type
            if self.entropies[0] == "Shannon":
                ce = (
                    -(y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12))
                    .sum(2)
                    .mean(1)
                    .sum(0)
                )
            elif self.entropies[0] == "Alpha":
                ce = torch.pow(y_s_one_hot, self.alpha_values[0]) * torch.pow(
                    logits_s.softmax(2) + 1e-12, 1 - self.alpha_values[0]
                )
                ce = ((1 - ce.sum(2)) / (self.alpha_values[0] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Marginal entropy type
            if self.entropies[1] == "Shannon":
                q_ent = -(q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
            elif self.entropies[1] == "Alpha":
                q_ent = (
                    (1 - (torch.pow(q_probs.mean(1), self.alpha_values[1])).sum(1))
                    / (self.alpha_values[1] - 1)
                ).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Conditional entropy type
            if self.entropies[2] == "Shannon":
                q_cond_ent = (
                    -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
                )
            elif self.entropies[2] == "Alpha":
                q_cond_ent = (
                    (
                        (1 - (torch.pow(q_probs + 1e-12, self.alpha_values[2])).sum(2))
                        / (self.alpha_values[2] - 1)
                    )
                    .mean(1)
                    .sum(0)
                )
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Loss
            loss = self.loss_weights[0] * ce - (
                self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t1 = time.time()
            criterions = self.get_criterions(weights_old)
            self.record_convergence(timestamp=t1 - t0, criterions=criterions)

        self.backbone.eval()
        self.record_acc(query=query, y_q=y_q)
