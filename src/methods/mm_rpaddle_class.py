import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.methods.utils import get_one_hot, simplex_project, Glasso_cov_estimate
from src.methods.abstract_method import AbstractMethod, MinMaxScaler
from src.methods.rpaddle_base import RPADDLE_base
from src.methods.mm_rapddle_id import MM_PADDLE_id


class MM_PADDLE_class(MM_PADDLE_id):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)

    def init_params(self, support, y_s, query):
        n_tasks, n_query, feature_dim = query.size()
        _, n_support, _ = support.size()

        self.init_prototypes(support, y_s)
        self.theta = torch.ones(n_tasks, n_support + n_query, self.n_class).to(
            self.device
        )
        self.theta[:, :n_support] = 1
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)

        if not self.id_cov:
            # self.q strores the n_class covariances
            self.q = (
                torch.eye(feature_dim)
                .unsqueeze(0)
                .repeat(n_tasks, self.n_class, 1, 1)
                .to(self.device)
            )

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
        feature_dim = samples_query.size(2)
        arg_softmax = (
            (
                (-1 / 2)
                * self.rho_beta(samples_query)
                / (self.theta[:, -n_query:] ** (self.beta - 1))
            )
            + ((self.lambd / n_query) * (torch.log(class_prop.unsqueeze(1))))
            - feature_dim * (1 - 1 / self.beta) * torch.log(self.theta[:, -n_query:])
        )
        self.u = arg_softmax.softmax(-1)

    def update_theta(self, samples, all_u):
        """
        Updates the theta parameters

        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Updates:
            self.theta : torch.Tensor of shape [n_task, n_support + n_query, n_class]

        Returns:
            None
        """
        _, _, feature_dim = samples.size()
        if not self.change_theta_reg:
            self.theta = (
                (self.beta - 1)
                / 2
                * 1
                / (feature_dim * (1 - 1 / self.beta) - self.kappa)
                * ((self.rho_beta(samples)))
            ) ** (1 / (self.beta - 1))
        else:
            self.theta = (
                (self.beta - 1)
                / 2
                * (self.eta**self.p)
                / self.p
                * ((self.rho_beta(samples)))
            ) ** (1 / (self.beta - 1))

    def update_prot(self, samples, all_u):
        n_tasks, n_samples, _ = samples.size()
        rho = self.rho(samples).unsqueeze(-1)
        den_den = (rho ** (1 - self.beta / 2)) * (self.theta.unsqueeze(3)) ** (
            self.beta - 1
        )
        num = ((all_u.unsqueeze(-1) * samples.unsqueeze(2)) / den_den).sum(dim=1)

        den = (all_u.unsqueeze(-1) / den_den).sum(dim=1)

        self.prototypes = num / den


class MM_RPADDLE_Class_Reg(MM_PADDLE_class):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.deg_reg_theta = args.deg_reg_theta
        if self.deg_reg_theta == 1:
            # self.reg_theta = (self.kappa/(self.args.shots*self.args.n_class_support + self.args.n_query))*1000
            self.reg_theta = self.kappa

    def update_theta(self, samples, all_u):

        _, _, feature_dim = samples.size()
        if self.beta == 2.0 and self.deg_reg_theta == 1:
            # TODO : SOlveur direct
            # equation to solve is a degree 2 polynomial
            c = ((1 - self.beta) / 2) * (self.rho_beta(samples))
            b = (feature_dim * (1 - 1 / self.beta) - self.kappa) * all_u
            if self.reg_theta == 0:
                self.theta = -c / b
                return
            a_pos = self.reg_theta
            # a_neg = -self.reg_theta

            # delta = b ** 2 - 4 * a_pos * c

            delta_pos = b**2 - 4 * a_pos * c
            # delta_neg = b**2 - 4 * a_neg * c
            theta_pos = ((-b + torch.sqrt(delta_pos)) / (2 * a_pos)) + 10e-14
            # Removed since regularizing towards 1 makes no sense if no covariance is estimated at the same time
            # theta_neg = ((-b + torch.sqrt(delta_neg)) / (2 * a_neg)) + 10e-14

            # self.theta = torch.where(theta_pos > 1, theta_pos, theta_neg)
            self.theta = theta_pos
