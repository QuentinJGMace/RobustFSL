import torch
import torch.nn.functional as F
import time
from src.methods.mm_rpaddle_reg import MM_RPADDLE_reg
from src.methods.mm_rapddle_id import MM_PADDLE_id


class MM_RPADDLE_reg_sigmas(MM_PADDLE_id):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.deg_reg_theta = args.deg_reg_theta
        if self.deg_reg_theta == 1:
            # self.reg_theta = (self.kappa/(self.args.shots*self.args.n_class_support + self.args.n_query))*1000
            self.reg_theta = self.kappa

        self.zeta = 1 / 10

    def init_params(self, support, y_s, query):
        n_task, _, _ = support.size()
        self.sigmas = torch.ones(n_task, self.n_class).to(self.device)
        return super().init_params(support, y_s, query)

    def rho(self, samples):
        """
        Computes the rho function
        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Returns:
            rho : torch Tensor of shape (n_task, n_samples, n_class)
        """
        dist_2 = (
            self.sigmas.unsqueeze(1).unsqueeze(3) * samples.unsqueeze(2)
            - self.prototypes.unsqueeze(1)
        ).norm(dim=-1) ** 2
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
            self.sigmas.unsqueeze(1).unsqueeze(3) * samples.unsqueeze(2)
            - self.prototypes.unsqueeze(1)
        ).norm(dim=-1) ** 2
        return (dist_2 + self.eps) ** (self.beta / 2)

    def update_theta(self, samples, all_u):

        _, _, feature_dim = samples.size()
        if self.beta == 2.0 and self.deg_reg_theta == 1:
            # TODO : SOlveur direct
            # equation to solve is a degree 2 polynomial
            c = (1 - self.beta) / 2 * (self.rho_beta(samples) * all_u).sum(2)
            b = feature_dim * (1 - 1 / self.beta) - self.kappa
            if self.reg_theta == 0:
                self.theta = -c / b
                return
            a = self.reg_theta

            delta = b**2 - 4 * a * c
            self.theta = (-b + torch.sqrt(delta)) / (2 * a)  # + 1
        elif self.beta == 1.5 and self.deg_reg_theta == 1:
            pass

    def update_sigma(self, samples, all_u):

        _, _, feature_dim = samples.size()
        den_sum = self.rho(samples) * self.theta.unsqueeze(2)
        num_den = (torch.norm(samples, dim=-1) ** 2).unsqueeze(2) * all_u
        num_num = (
            torch.einsum(
                "bncf,bncf->bnc", samples.unsqueeze(2), self.prototypes.unsqueeze(1)
            )
            * all_u
        )

        den = (self.beta * num_den / den_sum).sum(1)
        num = feature_dim * (all_u - self.zeta).sum(1) + self.beta * (
            num_num / den_sum
        ).sum(1)

        self.sigma = num / den

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
            self.update_sigma(all_samples, all_u)

            t_end = time.time()
            # if i == self.n_iter - 1:
            #     print(u_old.isnan().any(), self.u.isnan().any())
            criterions = self.get_criterions(prototypes_old, theta_old, u_old)
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        self.record_acc(y_q=y_q, indexes_outliers_query=idx_outliers_query)


class MM_RPADDLE_reg_diag(MM_PADDLE_id):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        if hasattr(args, "zeta"):
            self.zeta = args.zeta
        else:
            self.zeta = self.args.shots

    def init_params(self, support, y_s, query):
        n_task, _, feature_dim = support.size()
        self.sigmas = torch.ones(n_task, self.n_class, feature_dim).to(self.device)
        return super().init_params(support, y_s, query)

    def rho(self, samples):
        """
        Computes the rho function
        Args:
            samples : torch.Tensor of shape [n_task, n_support + n_query, feature_dim]

        Returns:
            rho : torch Tensor of shape (n_task, n_samples, n_class)
        """
        dist_2 = (
            self.sigmas.unsqueeze(1) * samples.unsqueeze(2)
            - self.prototypes.unsqueeze(1)
        ).norm(dim=-1) ** 2
        return dist_2 + self.eps

    def rho_beta(self, samples):
        dist_2 = (
            self.sigmas.unsqueeze(1) * samples.unsqueeze(2)
            - self.prototypes.unsqueeze(1)
        ).norm(dim=-1) ** 2
        return (dist_2 + self.eps) ** (self.beta / 2)

    def update_sigma(self, samples, all_u):

        _, _, feature_dim = samples.size()
        den_sum = self.rho(samples).unsqueeze(3) * self.theta.unsqueeze(2).unsqueeze(
            3
        ) ** (self.beta - 1)
        num_den = (samples**2).unsqueeze(2) * all_u.unsqueeze(3)
        num_num = (
            all_u.unsqueeze(3) * samples.unsqueeze(2) * self.prototypes.unsqueeze(1)
        )

        den = (self.beta * num_den / den_sum).sum(1)
        num = (
            (self.beta * num_num / den_sum).sum(1)
            - self.zeta
            + all_u.unsqueeze(3).sum(1)
        )

        self.sigmas = num / den

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
            self.update_sigma(all_samples, all_u)

            t_end = time.time()
            # if i == self.n_iter - 1:
            #     print(u_old.isnan().any(), self.u.isnan().any())
            criterions = self.get_criterions(prototypes_old, theta_old, u_old)
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        self.record_acc(y_q=y_q, indexes_outliers_query=idx_outliers_query)
