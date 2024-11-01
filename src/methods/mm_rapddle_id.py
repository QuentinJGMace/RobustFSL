import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.methods.utils import get_one_hot, simplex_project, Glasso_cov_estimate
from src.methods.abstract_method import AbstractMethod, MinMaxScaler
from src.methods.rpaddle_base import RPADDLE_base


class MM_PADDLE_id(RPADDLE_base):
    """MM-PADDLE with identity covariance matrix
    
    """
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        if self.kappa != 0:
            self.eta = (2 / self.kappa) ** (1 / 2)
        self.eps = 1e-12 # to avoid numerical instability
        self.temp = self.args.temp

        if hasattr(self.args, "threshold"):
            self.soft_threshold = True
            self.threshold = self.args.threshold
        else:
            self.soft_threshold = False

    def rho(self, samples):
        """
        Computes the rho function
        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Returns:
            rho : torch Tensor of shape (n_task, n_samples, n_class)
        """
        dist_2 = (
            self.temp
            * (samples.unsqueeze(2) - self.prototypes.unsqueeze(1)).norm(dim=-1) ** 2
        )
        return dist_2 + self.eps

    def rho_beta(self, samples):
        """
        Computes the rho_beta function
        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Returns:
            rho^(beta/2) : torch Tensor of shape (n_task, n_samples, n_class)
        """
        dist_2 = (
            self.temp
            * (samples.unsqueeze(2) - self.prototypes.unsqueeze(1)).norm(dim=-1) ** 2
        )
        return (dist_2 + self.eps) ** (self.beta / 2)

    def update_u(self, samples_query, class_prop):
        """
        Updates the soft labels u

        Args:
            samples : torch.Tensor of shape [n_task, n_query, feature_dim]
            class_prop : torch.Tensor of shape [n_task, n_class]

        Updates:
            self.u : torch.Tensor of shape [n_task, n_query, n_class]

        Returns:
            None
        """
        n_query = samples_query.size(1)
        arg_softmax = (
            (-1 / 2)
            * self.rho_beta(samples_query)
            / (self.theta.unsqueeze(2)[:, -n_query:] ** (self.beta - 1))
        ) + (self.lambd / n_query) * (torch.log(class_prop.unsqueeze(1)))
        self.u = arg_softmax.softmax(-1)

    def update_theta(self, samples, all_u):
        """
        Updates the theta parameters

        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Updates:
            self.theta : torch.Tensor of shape [n_task, n_support + n_query]

        Returns:
            None
        """
        _, _, feature_dim = samples.size()
        self.theta = (
            (self.beta - 1)
            / 2
            * 1
            / (feature_dim * (1 - 1 / self.beta) - self.kappa) # Kappa should be generally 0 here
            * ((all_u * self.rho_beta(samples)).sum(dim=-1))
        ) ** (1 / (self.beta - 1))

        # Applies soft thresholding if needed
        if self.soft_threshold:
            self.theta = torch.sign(self.theta) * (
                torch.maximum(
                    self.theta.abs() - self.threshold, torch.ones_like(self.theta)
                )
            )

    def update_prot(self, samples, all_u):
        """
        Updates the prototypes

        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Updates:
            self.prototypes : torch.Tensor of shape [n_task, n_class, feature_dim]

        Returns:
            None
        """
        n_tasks, n_samples, _ = samples.size()
        rho = self.rho(samples).unsqueeze(-1)
        den_den = (rho ** (1 - self.beta / 2)) * (
            self.theta.unsqueeze(2).unsqueeze(3)
        ) ** (self.beta - 1)
        num = ((all_u.unsqueeze(-1) * samples.unsqueeze(2)) / den_den).sum(dim=1)

        den = (all_u.unsqueeze(-1) / den_den).sum(dim=1)

        self.prototypes = num / den

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

        t0 = time.time()
        y_s_one_hot = F.one_hot(y_s, self.n_class).to(self.device)

        self.init_params(support, y_s, query)

        n_query = query.size(1)
        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        feature_dim = all_samples.size(-1)

        # Optimization loop
        for i in range(self.n_iter):

            prototypes_old = self.prototypes.clone()
            theta_old = self.theta.clone()
            u_old = self.u.clone()
            t0 = time.time()
            all_u = torch.cat([y_s_one_hot.float(), self.u.float()], dim=1)

            class_prop = self.u.mean(1)

            self.update_u(query, class_prop)
            self.update_theta(all_samples, all_u)
            self.update_prot(all_samples, all_u)

            t_end = time.time()

            criterions = self.get_criterions(prototypes_old, theta_old, u_old)
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        self.record_acc(y_q=y_q, indexes_outliers_query=idx_outliers_query)
