import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.methods.utils import get_one_hot
from src.methods.abstract_method import AbstractMethod, MinMaxScaler


class RPADDLE_base(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lambd = args.lambd
        self.lr = args.lr
        self.beta = args.beta
        self.kappa = args.kappa
        self.id_cov = args.id_cov
        if self.kappa != 0:
            self.eta = (2 / self.kappa) ** (1 / 2)
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
        self.prototypes = weights / counts

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
        self.theta = torch.ones(n_tasks, n_support + n_query).to(self.device)
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

    def get_logits(self, samples) -> torch.Tensor:
        # logits for sample n and class k is -1/2 (x_n - prototype_k)^T* (theta_n*Q_k)^2 * (x_n - prototype_k)
        n_query = samples.size(1)
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

        # Extract adaptation logs
        logs = self.get_logs()

        # Stores mults
        if self.args.save_mult_outlier:
            self.mults = {}
            tau = self.theta ** (self.beta / (self.beta - 1))
            if self.theta.size(1) == support.size(1) + query.size(1):
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
            else:
                self.mults["query"] = tau
                print(
                    torch.max(tau[:, :]),
                    torch.min(tau[:, :]),
                )

        # if self.args.plot:
        #     self.plot_convergence()

        return logs

    def run_method(self, support, query, y_s, y_q):
        """
        Runs the method
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
        """
        raise NotImplementedError("This should not be called as this class is abstract")

    def get_criterions(self, old_proto, old_theta, old_u, old_cov=None):
        """
        returns:
            criterions : dict {"proto": float, "theta": float, "u": float, (optional)"cov": float}
        """
        with torch.no_grad():
            crit_proto = (
                (
                    (self.prototypes - old_proto).norm(dim=[1, 2])
                    / old_proto.norm(dim=[1, 2])
                )
                .mean()
                .item()
            )
            if self.theta.ndim == 2:
                crit_theta = (
                    ((self.theta - old_theta).norm(dim=[1]) / old_theta.norm(dim=[1]))
                    .mean()
                    .item()
                )
            elif self.theta.ndim == 3:
                crit_theta = (
                    (
                        (self.theta - old_theta).norm(dim=[1, 2])
                        / old_theta.norm(dim=[1, 2])
                    )
                    .mean()
                    .item()
                )
            crit_u = (
                ((self.u - old_u).norm(dim=[1, 2]) / old_u.norm(dim=[1, 2]))
                .mean()
                .item()
            )
            if not self.id_cov:
                crit_cov = (
                    (
                        (self.q - old_cov).norm(dim=[1, 2, 3])
                        / old_cov.norm(dim=[1, 2, 3])
                    )
                    .mean()
                    .item()
                )

        dic = {
            "proto": crit_proto,
            "theta": crit_theta,
            "u": crit_u,
        }
        if not self.id_cov:
            dic["cov"] = crit_cov
        return dic
