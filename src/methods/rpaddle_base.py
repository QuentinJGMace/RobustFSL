import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.methods.utils import get_one_hot
from src.methods.abstract_method import AbstractMethod, MinMaxScaler

import matplotlib.pyplot as plt
from src.experiments.plot_utils import (
    plot_pca,
    load_pca,
    plot_outlier_detection,
    plot_outlier_detection_per_class,
)
import random


class RPADDLE_base(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lambd = args.lambd
        self.lr = args.lr
        self.beta = args.beta
        self.kappa = args.kappa
        self.id_cov = args.id_cov
        self.n_support = self.args.shots * self.args.n_class_support
        if self.kappa != 0:
            self.eta = (2 / self.kappa) ** (1 / 2)
        if hasattr(self.args, "temp"):
            self.temp = self.args.temp
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
        if self.temp:
            logits = (-1 / 2) * self.temp * dist
        else:
            logits = (-1 / 2) * dist

        return logits

    def predict(self):
        """
        returns:
            preds : torch.Tensor of shape [n_task, n_query]
        """
        with torch.no_grad():
            preds = self.u.argmax(2)

            # if self.args.save_mult_outlier:
            #     self.mults = {}
            #     tau = self.theta ** (self.beta / (self.beta - 1))
            #     if self.theta.size(1) == self.n_support + self.args.n_query:
            #         self.mults["support"] = tau[:, : self.n_support]
            #         self.mults["query"] = tau[:, self.n_support :]
            #         print(
            #             torch.max(tau[:, : self.n_support]),
            #             torch.min(tau[:, : self.n_support]),
            #         )
            #         print(
            #             torch.max(tau[:, self.n_support :]),
            #             torch.min(tau[:, self.n_support :]),
            #         )
            #     else:
            #         self.mults["query"] = tau
            #         print(
            #             torch.max(tau[:, :]),
            #             torch.min(tau[:, :]),
            #         )
        return preds

    def run_task(self, task_dic, shot):
        """
        Fits the model to the support set
        inputs:
            task_dic : dict {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int
        """
        support, query, y_s, y_q, x_mean, idx_outliers_support, idx_outliers_query = (
            task_dic["x_s"],
            task_dic["x_q"],
            task_dic["y_s"],
            task_dic["y_q"],
            task_dic["x_mean"],
            task_dic["outliers_support"],
            task_dic["outliers_query"],
        )

        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        x_mean = x_mean.to(self.device)
        idx_outliers_support = idx_outliers_support.to(self.device)
        idx_outliers_query = idx_outliers_query.to(self.device)

        # Perform normalizations
        support, query = self.normalizer(support, query, train_mean=x_mean)

        # Run adaptation
        self.run_method(
            support=support,
            query=query,
            y_s=y_s,
            y_q=y_q,
            idx_outliers_support=idx_outliers_support,
            idx_outliers_query=idx_outliers_query,
        )

        # pca = load_pca("random/pca/pca_mini.pkl")
        # fig, ax = plot_pca(pca, support[0].cpu().numpy(), y_s[0].cpu().numpy(), save_path=f"random/pca/pca_support_outliers_0_{random.randint(0, 100000)}.png", plot=None, return_fig=True, markersize=10)
        # fig, ax = plot_pca(pca, query[0].cpu().numpy(), y_q[0].cpu().numpy(), save_path=f"random/pca/pca_query_outliers_0_{random.randint(0, 100000)}.png", plot=(fig, ax), return_fig=True, markersize=10)
        # fig, ax = plot_pca(pca, self.prototypes[0].cpu().numpy(), None, save_path=f"random/pca/pca_with_mean_outliers_0_{random.randint(0, 100000)}.png", plot=(fig, ax), return_fig=True, markersize=50)

        # 1/0

        # if len(self.theta.size()) == 2:
        #     plot_outlier_detection(self.theta, self.n_support, idx_outliers_support, idx_outliers_query, save_path=f"random/outlier_detection/outliers_detection_outliers_ood32_support_thresholding_OOD_dirichlet_teeeeeeest_{random.randint(0, 100000)}.png", plot=None, return_fig=False)
        # elif len(self.theta.size()) == 3:
        #     plot_outlier_detection_per_class(self.theta, self.n_support, idx_outliers_support, idx_outliers_query, save_path=f"random/outlier_detection/outliers_detection_per_class_reg_outliers_ood32_{random.randint(0, 100000)}.png", plot=None, return_fig=False)
        # 1/0

        # Extract adaptation logs
        logs = self.get_logs()

        # Stores mults
        if self.args.save_mult_outlier:
            self.mults = {}
            tau = self.theta ** (self.beta / (self.beta - 1))
            if self.theta.size(1) == support.size(1) + query.size(1):
                self.mults["support"] = tau[:, : support.size(1)]
                self.mults["query"] = tau[:, support.size(1) :]
                with open("thetas_mm_red_curve.txt", "a") as f:
                    f.write(f"N shots : {self.args.shots}" + "\n")
                    f.write(
                        f"N outliers support : {self.args.n_outliers_support}, N outliers query : {self.args.n_outliers_query}"
                        + "\n"
                    )
                    f.write(
                        f"Support mean: {self.theta[:, : support.size(1)].nanmean().item()} Query : {self.theta[:, support.size(1) :].nanmean().item()}"
                        + "\n"
                    )
                    f.write(
                        f"Support max: {self.theta[:, : support.size(1)][~self.theta[:, : support.size(1)].isnan()].max().item()} Query : {self.theta[:, support.size(1) :][~self.theta[:, support.size(1) :].isnan()].max().item()}"
                        + "\n"
                    )
                    f.write(
                        f"Support min: {self.theta[:, : support.size(1)][~self.theta[:, : support.size(1)].isnan()].min().item()} Query : {self.theta[:, support.size(1) :][~self.theta[:, support.size(1) :].isnan()].min().item()}"
                        + "\n"
                    )
                    f.write(
                        f"Number of nans support : {self.theta[:, : support.size(1)].isnan().sum().item()} Query : {self.theta[:, support.size(1) :].isnan().sum().item()}"
                        + "\n"
                    )
                    f.write(
                        f"Number of infs support : {self.theta[:, : support.size(1)].isinf().sum().item()} Query : {self.theta[:, support.size(1) :].isinf().sum().item()}"
                        + "\n"
                    )
                    f.write(
                        f"Number of zeros support : {self.theta[:, : support.size(1)].eq(0).sum().item()} Query : {self.theta[:, support.size(1) :].eq(0).sum().item()}"
                        + "\n"
                    )
                    f.write("-------------------------------------------" + "\n\n")
                # print(
                #     torch.max(tau[:, : support.size(1)]),
                #     torch.min(tau[:, : support.size(1)]),
                # )
                # print(
                #     torch.max(tau[:, support.size(1) :]),
                #     torch.min(tau[:, support.size(1) :]),
                # )
            else:
                self.mults["query"] = tau
                print(
                    torch.max(tau[:, :]),
                    torch.min(tau[:, :]),
                )

        # if self.args.plot:
        #     self.plot_convergence()

        return logs

    def run_method(
        self, support, query, y_s, y_q, idx_outliers_support, idx_outliers_query
    ):
        """
        Runs the method
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
            idx_outliers_support : torch.Tensor of shape [n_task, n_outliers_support]
            idx_outliers_query : torch.Tensor of shape [n_task, n_outliers_query]
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
                .nanmean()
                .item()
            )
            if self.theta.ndim == 2:
                crit_theta = (
                    ((self.theta - old_theta).norm(dim=[1]) / old_theta.norm(dim=[1]))
                    .nanmean()
                    .item()
                )
            elif self.theta.ndim == 3:
                crit_theta = (
                    (
                        (self.theta - old_theta).norm(dim=[1, 2])
                        / old_theta.norm(dim=[1, 2])
                    )
                    .nanmean()
                    .item()
                )
            crit_u = (
                ((self.u - old_u).norm(dim=[1, 2]) / old_u.norm(dim=[1, 2]))
                .nanmean()
                .item()
            )
            if not self.id_cov:
                crit_cov = (
                    (
                        (self.q - old_cov).norm(dim=[1, 2, 3])
                        / old_cov.norm(dim=[1, 2, 3])
                    )
                    .nanmean()
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
