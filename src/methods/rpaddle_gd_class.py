"""Rapaddle implementation with SGD, theta also varies on the support set contrary to the original implementation"""

import time

import torch
from tqdm import tqdm
import torch.nn.functional as F
from src.methods.utils import get_one_hot, simplex_project
from src.methods.rpaddle_base import RPADDLE_base


class MultNoisePaddle_GD2_class(RPADDLE_base):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)

    def init_params(self, support, y_s, query):
        n_tasks, n_query, feature_dim = query.size()
        _, n_support, _ = support.size()

        self.init_prototypes(support, y_s)
        self.theta = torch.ones(n_tasks, n_support + n_query, self.n_class).to(
            self.device
        )
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)

        if not self.id_cov:
            # self.q strores the n_class covariances
            self.q = (
                torch.eye(feature_dim)
                .unsqueeze(0)
                .repeat(n_tasks, self.n_class, 1, 1)
                .to(self.device)
            )

            self.q = torch.nn.Parameter(self.q)

        self.prototypes = torch.nn.Parameter(self.prototypes)
        self.theta = torch.nn.Parameter(self.theta)
        self.u = torch.nn.Parameter(self.u)

    def compute_mahalanobis(self, samples, theta):
        """
        Computes the Mahalanobis distance between the samples and the prototypes
        inputs:
            samples : torch.Tensor of shape [n_task, n_sample, feature_dim]
            theta : torch.Tensor of shape [n_task, n_sample, n_class]
        """
        if not self.id_cov:
            tranformed_samples = torch.matmul(
                self.q.unsqueeze(1), samples.unsqueeze(2).unsqueeze(-1)
            ).squeeze(
                -1
            )  # Q_kx_n [n_task, n_sample, n_class, feature_dim]
        else:
            tranformed_samples = samples.unsqueeze(2)
        # computes the norm of the difference between the samples and the prototypes
        dist_beta = (tranformed_samples - self.prototypes.unsqueeze(1)).norm(
            dim=-1
        ) ** self.beta  # ||Q_kx_n - prototype_k||^beta [n_task, n_sample, n_class]
        dist_beta /= theta ** (
            self.beta - 1
        )  # ||Q_kx_n - prototype_k||^beta/theta [n_task, n_sample, n_class]

        return dist_beta

    def run_method(self, support, query, y_s, y_q):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
        """
        # self.logger.info(
        #     " ==> Executing RobustPADDLE with LAMBDA = {}".format(self.lambd)
        # )

        t0 = time.time()
        y_s_one_hot = F.one_hot(y_s, self.n_class).to(self.device)

        self.init_params(support, y_s, query)
        if not self.id_cov:
            optimizer = torch.optim.Adam(
                [self.prototypes, self.theta, self.u, self.q], lr=self.lr
            )
        else:
            optimizer = torch.optim.Adam(
                [self.prototypes, self.theta, self.u], lr=self.lr
            )

        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        feature_dim = all_samples.size(-1)

        n_tasks = query.size(0)

        losses = []
        for i in tqdm(range(self.n_iter)):

            prototypes_old = self.prototypes.detach().clone()
            theta_old = self.theta.detach().clone()
            u_old = self.u.detach().clone()

            if not self.id_cov:
                q_old = self.q.detach().clone()
            else:
                q_old = None
            t0 = time.time()

            # Data fitting term
            distances = self.compute_mahalanobis(all_samples, self.theta)
            all_p = torch.cat([y_s_one_hot.float(), self.u.float()], dim=1)

            data_fitting = (
                (distances * all_p).sum((-2, -1)).mean(0)
            )  # .mean(0) for more stability ?

            # Theta regularization term
            sum_log_theta = (
                (
                    all_p
                    * (feature_dim * (1 - 1 / self.beta) - self.kappa)
                    * torch.log(self.theta + 1e-12)
                )
                .sum(dim=(1, 2))
                .mean(0)
            )
            l2_theta = (
                1
                / self.n_class
                * (1 / self.eta)
                * (self.theta.reshape(n_tasks, -1).norm(dim=-1, keepdim=False, p=2))
                ** 2
            ).mean(0)

            theta_term = sum_log_theta + l2_theta

            # Covariance regularizing term
            if not self.id_cov:
                q_log_det = torch.logdet(self.q).unsqueeze(1)
                q_term = (all_p * q_log_det).sum((-2, -1)).mean(0)
            else:
                q_term = 0

            # partition complexity term
            query_class_ratios = self.u.mean(1).to(self.device)
            partition_complexity = (
                (query_class_ratios * torch.log(query_class_ratios + 1e-12))
                .sum(-1)
                .mean(0)
            )

            # enthropic barrier
            ent_barrier = (self.u * torch.log(self.u + 1e-12)).sum(-1).sum(-1).mean(0)

            # final loss
            loss = (
                data_fitting
                - self.lambd * partition_complexity
                + theta_term
                - q_term
                # + ent_barrier  # - self.lambd * partition_complexity + theta_term - q_term
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Projection into simplex
            with torch.no_grad():
                self.u.data = simplex_project(self.u, device=self.device)

            t_end = time.time()
            criterions = self.get_criterions(
                old_proto=prototypes_old,
                old_theta=theta_old,
                old_u=u_old,
                old_cov=q_old,
            )
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        # plt.plot(losses)
        # plt.savefig("losses.png")
        self.record_acc(y_q=y_q)
