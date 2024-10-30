"""RPADDLE implementation with SGD, theta also varies on the support set contrary to the original implementation"""

import time

import torch
import torch.nn.functional as F
from src.methods.utils import get_one_hot, simplex_project
from src.methods.rpaddle_base import RPADDLE_base

# import matplotlib.pyplot as plt
# import random


class MultNoisePaddle_GD(RPADDLE_base):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.optimizer = args.optimizer

    def init_params(self, support, y_s, query):
        super().init_params(support, y_s, query)
        self.theta = torch.nn.Parameter(self.theta)
        self.u = torch.nn.Parameter(self.u)
        self.prototypes = torch.nn.Parameter(self.prototypes)
        if not self.id_cov:
            self.q = torch.nn.Parameter(self.q)

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
        # self.logger.info(
        #     " ==> Executing RobustPADDLE with LAMBDA = {}".format(self.lambd)
        # )

        t0 = time.time()
        y_s_one_hot = F.one_hot(y_s, self.n_class).to(self.device)

        # Init the parameters (namely prototypes, theta, u)
        self.init_params(support, y_s, query)

        # Optimizer definiton
        if not self.id_cov:
            # We should not try to optimize q as it makes for too many parameters
            learnable_params = [self.prototypes, self.theta, self.u, self.q]
        else:
            learnable_params = [self.prototypes, self.theta, self.u]

        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(learnable_params, lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(learnable_params, lr=self.lr)

        # Concatenates support and query to have them in a signle vector
        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        feature_dim = all_samples.size(-1)

        losses = []
        # Optimisation loop
        for i in range(self.n_iter):

            prototypes_old = self.prototypes.detach().clone()
            theta_old = self.theta.detach().clone()
            u_old = self.u.detach().clone()

            if not self.id_cov:
                q_old = self.q.detach().clone()
            else:
                q_old = None
            t0 = time.time()

            # Loss computation
            # Data fitting term
            distances = self.compute_mahalanobis(all_samples, self.theta)
            all_p = torch.cat([y_s_one_hot.float(), self.u.float()], dim=1)

            data_fitting = (
                (distances * all_p).sum((-2, -1)).sum(0)
            )

            # Theta regularization term
            sum_log_theta = (
                feature_dim * (1 - 1 / self.beta) - self.kappa
            ) * torch.log(self.theta + 1e-12).sum(1).sum(0)
            if self.kappa != 0:
                l2_theta = (
                    (1 / self.eta) * ((self.theta - 1).norm(dim=-1, keepdim=False, p=1))
                ).sum(0)
            else:
                l2_theta = 0

            theta_term = sum_log_theta + l2_theta

            # Covariance regularizing term
            if not self.id_cov:
                q_log_det = torch.logdet(self.q).unsqueeze(1)
                q_term = (all_p * q_log_det).sum((-2, -1)).sum(0)
            else:
                q_term = 0

            # partition complexity term
            query_class_ratios = self.u.mean(1).to(self.device)
            partition_complexity = (
                (query_class_ratios * torch.log(query_class_ratios + 1e-12))
                .sum(-1)
                .sum(0)
            )

            # enthropic barrier
            ent_barrier = (self.u * torch.log(self.u + 1e-12)).sum(-1).sum(-1).sum(0)

            # final loss
            loss = (
                data_fitting
                - self.lambd * partition_complexity
                + theta_term
                - q_term # This term should generally be 0, just here for consistency
                + ent_barrier
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Projection into simplex
            with torch.no_grad():
                self.u.data = simplex_project(self.u, device=self.device)

            # Register the losses and criterions
            t_end = time.time()
            criterions = self.get_criterions(
                old_proto=prototypes_old,
                old_theta=theta_old,
                old_u=u_old,
                old_cov=q_old,
            )
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        ## Plotting to debug uncomment if needed, whatch out not to plot thousands of figures
        # plt.figure(random.randint(1, 100000))
        # plt.scatter([i for i in range(len(losses))], losses)
        # plt.savefig(f"losses_{random.randint(1, 100000)}.png")
        self.record_acc(y_q=y_q, indexes_outliers_query=idx_outliers_query)
