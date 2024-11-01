import torch
from src.methods.mm_rapddle_id import MM_PADDLE_id


class MM_RPADDLE_reg(MM_PADDLE_id):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.deg_reg_theta = args.deg_reg_theta
        if self.deg_reg_theta == 1:
            # self.reg_theta = (self.kappa/(self.args.shots*self.args.n_class_support + self.args.n_query))*1000
            self.reg_theta = self.kappa

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

    def update_theta(self, samples, all_u):

        _, _, feature_dim = samples.size()
        if self.beta == 2.0 and self.deg_reg_theta == 1:
            # TODO : SOlveur direct
            # equation to solve is a degree 2 polynomial
            c = ((1 - self.beta) / 2) * (self.rho_beta(samples) * all_u).sum(2)
            b = feature_dim * (1 - 1 / self.beta) - self.kappa
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
        elif self.beta == 1.5 and self.deg_reg_theta == 1:
            pass
