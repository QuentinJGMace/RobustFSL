"""Rapaddle implementation with SGD, theta also varies on the support set contrary to the original implementation"""

import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.methods.utils import get_one_hot, simplex_project
from src.methods.abstract_method import AbstractMethod, MinMaxScaler


class MutlNoisePaddle_GD2(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lambd = args.lambd
        self.lr = args.lr
        self.beta = args.beta
        self.kappa = args.kappa
        self.id_cov = args.id_cov
        self.alpha = args.p
        self.eta = (self.alpha / self.kappa) ** (1 / self.alpha)
        self.init_info_lists()

    def init_prototypes(self, support, y_s):
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
        self.prototypes = torch.nn.Parameter(weights / counts)

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

        self.init_prototypes(support, y_s)
        self.theta = torch.nn.Parameter(
            torch.ones(n_tasks, n_query + n_support).to(self.device)
        )
        # self.q strores the n_class covariances
        self.q = torch.nn.Parameter(
            (
                torch.eye(feature_dim)
                .unsqueeze(0)
                .repeat(n_tasks, self.n_class, 1, 1)
                .to(self.device)
            )
        )
        self.u = torch.nn.Parameter(
            (self.get_logits(query)).softmax(-1).to(self.device)
        )

        self.theta.requires_grad = True
        self.u.requires_grad = True
        self.q.requires_grad = True

    def compute_mahalanobis(self, samples, theta):
        """
        Computes the Mahalanobis distance between the samples and the prototypes
        inputs:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]
        """
        # Computes the Mahalanobis distance between the samples and the prototypes
        # [n_task, n_sample, n_class]
        # self.q : [n_task, 1, n_class, feature_dim, feature_dim]
        # self.prototypes: [n_task, n_class, feature_dim]
        # samples : [n_task, n_sample, 1, feature_dim]
        tranformed_samples = torch.matmul(
            self.q.unsqueeze(1), samples.unsqueeze(2).unsqueeze(-1)
        ).squeeze(
            -1
        )  # Q_kx_n [n_task, n_sample, n_class, feature_dim]
        reshaped_theta = theta.unsqueeze(-1)  # theta shape : [n_task, n_sample, 1]
        # computes the norm of the difference between the samples and the prototypes
        dist_beta = (tranformed_samples - self.prototypes.unsqueeze(1)).norm(
            dim=-1
        ) ** self.beta  # ||Q_kx_n - prototype_k||^beta [n_task, n_sample, n_class]
        dist_beta /= reshaped_theta ** (
            self.beta - 1
        )  # ||Q_kx_n - prototype_k||^beta/theta [n_task, n_sample, n_class]

        return dist_beta

    def get_logits(self, samples) -> torch.Tensor:
        n_tasks, n_query = samples.size(0), samples.size(1)
        # logits for sample n and class k is -1/2 (x_n - prototype_k)^T* (theta_n*Q_k)^2 * (x_n - prototype_k)

        dist = self.compute_mahalanobis(samples, self.theta[:, -n_query:])
        logits = -1 / 2 * dist

        return logits

    def predict(self):
        """
        returns:
            preds : torch.Tensor of shape [n_task, n_query]
        """
        with torch.no_grad():
            preds = self.u.argmax(2)
        return preds

    def run_task(self, task_dic, shot):
        """
        Fits the model to the support set
        inputs:
            task_dic : dict {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int
        """
        support, query, y_s, y_q = (
            task_dic["x_s"],
            task_dic["x_q"],
            task_dic["y_s"],
            task_dic["y_q"],
        )

        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations
        scaler = MinMaxScaler(feature_range=(0, 1))
        query, support = scaler(query, support)

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # print(self.q[0, 0].diag())
        # print("----------------------------")
        # print(self.theta)

        # Extract adaptation logs
        logs = self.get_logs()

        # Stores mults
        if self.args.save_mult_outlier:
            self.mults = {}
            tau = self.theta ** (self.beta / (self.beta - 1))
            self.mults["support"] = tau[:, : support.size(1)]
            self.mults["query"] = tau[:, support.size(1) :]

            print(torch.max(self.theta))

        # if self.args.plot:
        #     self.plot_convergence()

        return logs

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
        optimizer = torch.optim.AdamW(
            [self.prototypes, self.theta, self.u, self.q], lr=self.lr
        )

        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        feature_dim = all_samples.size(-1)

        losses = []
        for i in range(self.n_iter):

            prototypes_old = self.prototypes.detach().clone()
            theta_old = self.theta.detach().clone()
            q_old = self.q.detach().clone()
            u_old = self.u.detach().clone()
            t0 = time.time()
            # all_theta = torch.cat([theta_support, self.theta], 1)

            # Data fitting term
            distances = self.compute_mahalanobis(all_samples, self.theta)
            all_p = torch.cat([y_s_one_hot.float(), self.u.float()], dim=1)

            data_fitting = (
                (distances * all_p).sum((-2, -1)).mean(0)
            )  # .mean(0) for more stability ?

            # Theta regularization term
            sum_log_theta = (
                feature_dim * (1 - 1 / self.beta) - self.kappa
            ) * torch.log(self.theta + 1e-12).sum(1).mean(0)
            l2_theta = (
                ((1 / self.eta) ** self.alpha)
                * (self.theta.norm(dim=-1, keepdim=False, p=self.alpha)) ** self.alpha
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
            # ent_barrier = (self.u * torch.log(self.u + 1e-12)).sum(-1).sum(-1).mean(0)

            # final loss
            loss = (
                data_fitting
                - self.lambd * partition_complexity
                + theta_term
                # - q_term
                # + ent_barrier  # - self.lambd * partition_complexity + theta_term - q_term
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # print((prototypes_old - self.prototypes).norm(dim=-1))
            # print((theta_old - self.theta).norm(dim=-1))
            # print((q_old - self.q).norm(dim=-1))
            # print((u_old - self.u).norm(dim=-1))
            # print('----------------------------')
            # Projection into simplex
            with torch.no_grad():
                self.u.data = simplex_project(self.u, device=self.device)

            t_end = time.time()
            criterions = self.get_criterions(prototypes_old, q_old, theta_old, u_old)
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        # plt.plot(losses)
        # plt.savefig("losses.png")
        self.record_acc(y_q=y_q)

    def get_criterions(self, old_proto, old_cov, old_theta, old_u):
        """
        inputs:
            old_prot : torch.Tensor of shape [n_task, num_class, feature_dim]
            old_cov : torch.Tensor of shape [n_task, num_class, feature_dim, feature_dim]
            old_theta : torch.Tensor of shape [n_task, n_sample]
            old_u: torch.Tensor of shape [n_task, n_query, num_class]

        returns:
            criterions : torch.Tensor of shape [n_task]
        """
        with torch.no_grad():
            crit_proto = (self.prototypes - old_proto).norm(dim=[1, 2]).mean().item()
            crit_cov = (self.cov - old_cov).norm(dim=[1, 2, 3]).mean().item()
            crit_theta = (self.theta - old_theta).norm(dim=[1]).mean().item()
            crit_u = (self.u - old_u).norm(dim=[1, 2]).mean().item()

        return {
            "proto": crit_proto,
            "cov": crit_cov,
            "theta": crit_theta,
            "u": crit_u,
        }


class MutlNoisePaddle_GD_id2(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lambd = args.lambd
        self.lr = args.lr
        self.beta = args.beta
        self.kappa = args.kappa
        self.eta = (2 / self.kappa) ** (1 / 2)
        self.id_cov = args.id_cov
        self.init_info_lists()

    def init_prototypes(self, support, y_s):
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
        self.prototypes = torch.nn.Parameter(weights / counts)

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

        self.init_prototypes(support, y_s)
        self.theta = torch.nn.Parameter(
            torch.ones(n_tasks, n_support + n_query).to(self.device)
        )
        self.u = torch.nn.Parameter(
            (self.get_logits(query)).softmax(-1).to(self.device)
        )

        self.theta.requires_grad = True
        self.u.requires_grad = True
        self.prototypes.requires_grad = True

    def compute_mahalanobis(self, samples, theta):
        """
        Computes the Mahalanobis distance between the samples and the prototypes
        inputs:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]
        """
        # Computes the Mahalanobis distance between the samples and the prototypes
        # [n_task, n_sample, n_class]
        # self.q : [n_task, 1, n_class, feature_dim, feature_dim]
        # self.prototypes: [n_task, n_class, feature_dim]
        # samples : [n_task, n_sample, 1, feature_dim]
        tranformed_samples = samples.unsqueeze(
            2
        )  # Q_kx_n [n_task, n_sample, n_class, feature_dim]
        reshaped_theta = theta.unsqueeze(-1)  # theta shape : [n_task, n_sample, 1]
        # computes the norm of the difference between the samples and the prototypes
        dist_beta = (tranformed_samples - self.prototypes.unsqueeze(1)).norm(
            dim=-1
        ) ** self.beta  # ||Q_kx_n - prototype_k||^beta [n_task, n_sample, n_class]
        dist_beta /= reshaped_theta ** (
            self.beta - 1
        )  # ||Q_kx_n - prototype_k||^beta/theta [n_task, n_sample, n_class]

        return dist_beta

    def get_logits(self, samples) -> torch.Tensor:
        n_tasks, n_query = samples.size(0), samples.size(1)
        # logits for sample n and class k is -1/2 (x_n - prototype_k)^T* (theta_n*Q_k)^2 * (x_n - prototype_k)

        dist = self.compute_mahalanobis(samples, self.theta[:, -n_query:])
        logits = -1 / 2 * dist

        return logits

    def predict(self):
        """
        returns:
            preds : torch.Tensor of shape [n_task, n_query]
        """
        with torch.no_grad():
            preds = self.u.argmax(2)
        return preds

    def run_task(self, task_dic, shot):
        """
        Fits the model to the support set
        inputs:
            task_dic : dict {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int
        """
        support, query, y_s, y_q = (
            task_dic["x_s"],
            task_dic["x_q"],
            task_dic["y_s"],
            task_dic["y_q"],
        )

        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations
        scaler = MinMaxScaler(feature_range=(0, 1))
        # support = F.normalize(support, dim=2).to(self.device)
        # query = F.normalize(query, dim=2).to(self.device)
        query, support = scaler(query, support)

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Stores mults
        if self.args.save_mult_outlier:
            self.mults = {}
            tau = self.theta ** (self.beta / (self.beta - 1))
            self.mults["support"] = tau[:, : support.size(1)]
            self.mults["query"] = tau[:, support.size(1) :]

            print(
                torch.max(tau[:, : support.size(1)]),
                torch.min(tau[:, : support.size(1)]),
            )
            print(
                torch.max(tau[:, support.size(1) :]),
                torch.min(tau[:, support.size(1) :]),
            )

        # Extract adaptation logs
        logs = self.get_logs()

        return logs

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
        optimizer = torch.optim.SGD([self.prototypes, self.theta, self.u], lr=self.lr)

        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        feature_dim = all_samples.size(-1)
        losses = []
        for i in range(self.n_iter):

            prototypes_old = self.prototypes.clone()
            theta_old = self.theta.clone()
            u_old = self.u.clone()
            t0 = time.time()

            # Data fitting term
            distances = self.compute_mahalanobis(all_samples, self.theta)
            all_p = torch.cat([y_s_one_hot.float(), self.u.float()], dim=1)

            data_fitting = (
                (distances * all_p).sum((-2, -1)).mean(0)
            )  # .mean(0) for more stability ?

            # Theta regularization term
            sum_log_theta = (
                feature_dim * (1 - 1 / self.beta) - self.kappa
            ) * torch.log(self.theta + 1e-12).sum(1).mean(0)
            l2_theta = (
                1 / self.eta * (self.theta.norm(dim=-1, keepdim=False)) ** 2
            ).mean(0)

            theta_term = sum_log_theta + l2_theta

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
                + ent_barrier  # - self.lambd * partition_complexity + theta_term - q_term
            )
            losses.append(loss.mean().item())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Projection into simplex
            with torch.no_grad():
                self.u.data = simplex_project(self.u, device=self.device)

            t_end = time.time()
            criterions = self.get_criterions(prototypes_old, theta_old, u_old)
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        self.record_acc(y_q=y_q)

    def get_criterions(self, old_proto, old_theta, old_u):
        """
        inputs:
            old_prot : torch.Tensor of shape [n_task, num_class, feature_dim]
            old_theta : torch.Tensor of shape [n_task, n_sample]
            old_u: torch.Tensor of shape [n_task, n_query, num_class]

        returns:
            criterions : torch.Tensor of shape [n_task]
        """
        with torch.no_grad():
            crit_proto = (self.prototypes - old_proto).norm(dim=[1, 2]).mean().item()
            crit_theta = (self.theta - old_theta).norm(dim=[1]).mean().item()
            crit_u = (self.u - old_u).norm(dim=[1, 2]).mean().item()

        return {
            "proto": crit_proto,
            "theta": crit_theta,
            "u": crit_u,
        }
