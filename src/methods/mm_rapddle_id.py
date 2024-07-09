import time

import torch
import torch.nn.functional as F
from tqdm import tqdm
from src.methods.utils import get_one_hot, simplex_project
from src.methods.abstract_method import AbstractMethod, MinMaxScaler


class MM_PADDLE_id(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lambd = args.lambd
        self.lr = args.lr
        self.beta = args.beta
        self.kappa = args.kappa
        self.eta = (2 / self.kappa) ** (1 / 2)
        self.change_theta_reg = args.change_theta_reg
        if self.change_theta_reg:
            self.p = args.p
        self.id_cov = True
        self.eps = 1e-6
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
        self.theta = torch.ones(n_tasks, n_query + n_support).to(self.device)
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)

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

    def rho(self, samples):
        """
        Computes the rho function
        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Returns:
            rho : torch Tensor of shape (n_task, n_samples, n_class)
        """
        dist_2 = (samples.unsqueeze(2) - self.prototypes.unsqueeze(1)).norm(dim=-1) ** 2
        return dist_2 + self.eps

    def rho_beta(self, samples):
        """
        Computes the rho_beta function
        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Returns:
            rho^(beta/2) : torch Tensor of shape (n_task, n_samples, n_class)
        """
        dist_2 = (samples.unsqueeze(2) - self.prototypes.unsqueeze(1)).norm(dim=-1) ** 2
        return (dist_2 + self.eps) ** self.beta

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
        if not self.change_theta_reg:
            self.theta = (
                (self.beta - 1)
                / 2
                * 1
                / (feature_dim * (1 - 1 / self.beta) - self.kappa)
                * ((all_u * self.rho_beta(samples)).sum(dim=-1))
            ) ** (1 / (self.beta - 1))
        else:
            self.theta = (
                (self.beta - 1)
                / 2
                * (self.eta**self.p)
                / self.p
                * ((all_u * self.rho_beta(samples)).sum(dim=-1))
            ) ** (1 / (self.beta - 1))

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

        # Stores mults
        if self.args.save_mult_outlier:
            self.mults = {}
            tau = self.theta ** (self.beta / (self.beta - 1))
            self.mults["support"] = tau[:, : support.size(1)]
            self.mults["query"] = tau[:, support.size(1) :]

            print(torch.max(self.theta))

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

        n_query = query.size(1)
        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        feature_dim = all_samples.size(-1)
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
