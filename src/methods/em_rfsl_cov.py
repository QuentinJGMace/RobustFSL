from scipy.optimize import fsolve
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.methods.utils import get_one_hot
from src.methods.abstract_method import AbstractMethod, MinMaxScaler


class EM_RobustPaddle(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):

        super().__init__(backbone, device, log_file, args)
        self.beta = args.beta
        self.p = args.p
        self.gamma = args.gamma_prox
        self.kappa = args.kappa
        self.eta = (self.p / self.kappa) ** (1 / self.p)
        self.auto_init = args.auto_init  # auto init of hyperparams
        self.reg_q = args.reg_q
        self.zeta1 = args.zeta1
        self.zeta2 = args.zeta2
        self.zeta3 = args.omega1 * self.zeta1
        self.zeta4 = args.omega2 * self.zeta2

        self.init_info_lists()

    def init_class_prop(self, y_s):
        """
        Initializes the class proportions

        Args:
            y_s : one hot encoded tensor of shape [n_tasks, n_support, n_class]

        Returns:
            torch.Tensor of shape [n_tasks, n_class]
        """

        _, n_support, _ = y_s.size()
        class_prop = y_s.sum(1).float() / n_support
        return class_prop.to(self.device)

    def init_prototypes(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : one hot encoded torch.Tensor of shape [n_task, shot, n_class]

        Returns:
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        counts = y_s.sum(1).view(n_tasks, -1, 1)
        weights = y_s.transpose(1, 2).matmul(support)

        return (weights / counts).to(self.device)

    def init_hyperparams(self, samples):
        """
        Initializes the hyperparameters

        Args:
            samples : torch.Tensor of shape [n_tasks, n_sample, feature_dim]
        """
        n_tasks, n_sample, feature_dim = samples.size()
        normay = torch.sqrt(torch.median(torch.sum(samples**2, dim=2)))
        NoL2 = (
            ((torch.linalg.matrix_norm(samples, ord=2)).mean()) ** 2 / (normay**2)
            + n_sample
            + 1
        )
        self.gamma = (1 / torch.sqrt(NoL2)).cpu().item()
        self.zeta1 = (0.95 / (NoL2 * self.gamma)).cpu().item()
        self.zeta3 = (NoL2 * self.zeta1 / 2).cpu().item()
        self.zeta4 = self.zeta3
        self.zeta2 = 0.1 * 0.95 / self.gamma

    def init_params(self, support, y_s, query):
        """
        Initializes the parameters

        Args:
            support : torch.Tensor of shape [n_tasks, n_support, feature_dim]
            y_s : one hot encoded tensor of shape [n_tasks, n_support, n_class]
            query : torch.Tensor of shape [n_tasks, n_query, feature_dim]
        """
        n_tasks, n_support, feature_dim = support.size()
        n_tasks, n_query, feature_dim = query.size()
        self.n_support = n_support
        self.n_query = n_query

        n_sample = n_support + n_query

        self.class_prop = self.init_class_prop(y_s)
        self.cov = (
            torch.eye(feature_dim)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(n_tasks, self.n_class, 1, 1)
            .to(self.device)
        )
        self.prototypes = self.init_prototypes(support, y_s)
        self.theta = torch.ones(n_tasks, n_sample, self.n_class).to(self.device)

    def predict(self, samples):
        """
        Predicts the labels

        Returns:
            torch.Tensor of shape [n_tasks, n_query]
        """
        return self.compute_responsabilities(samples, self.n_support).argmax(2)

    def record_acc(self, y_q, preds_q):
        """
        Records the accuracy

        Args:
            y_q : torch.Tensor of shape [n_tasks, n_query]
        """
        accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
        self.test_acc.append(accuracy)

    def compute_distance_to_prototypes(self, samples):
        """
        Computes the distance to the prototypes

        Args:
            samples : torch.Tensor of shape [n_tasks, n_sample, feature_dim]

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class]
        """
        n_tasks, n_sample, feature_dim = samples.size()
        prototypes = self.prototypes.unsqueeze(1).expand(
            n_tasks, n_sample, self.n_class, feature_dim
        )
        samples = samples.unsqueeze(2).expand(
            n_tasks, n_sample, self.n_class, feature_dim
        )
        warped_samples = torch.matmul(
            self.cov.unsqueeze(1), samples.unsqueeze(-1)
        ).squeeze(-1)
        distance_to_prototypes = torch.norm(warped_samples - prototypes, dim=3)
        assert distance_to_prototypes.shape == (n_tasks, n_sample, self.n_class)
        return distance_to_prototypes

    def compute_log_density(self, beta_to_prototypes, n_support):
        """
        Computes the densities

        Args:
            beta_to_prototypes : torch.Tensor of shape [n_tasks, n_sample, n_class]

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class]
        """
        feature_dim = self.prototypes.size(2)

        exponant = (
            -1
            * beta_to_prototypes
            / (2 * torch.pow(self.theta[:, n_support:], self.beta - 1))
        )
        constant_term = (
            -1
            * feature_dim
            * ((self.beta - 1) / self.beta)
            * torch.log(self.theta[:, n_support:])
        )
        # print(exponant.mean())
        # print(constant_term.mean())
        # print(torch.log(self.theta).mean())
        # print('--------------------')
        return exponant + constant_term

    def compute_densities(self, beta_to_prototypes, n_support):
        """
        Computes the densities

        Args:
            beta_to_prototypes : torch.Tensor of shape [n_tasks, n_sample, n_class]

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class]
        """
        return torch.exp(self.compute_log_density(beta_to_prototypes, n_support))

    def compute_responsabilities(self, samples, n_support):
        """
        Computes the responsabilities

        Args:
            samples : torch.Tensor of shape [n_tasks, n_sample, n_class]

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class]
        """
        beta_to_prototypes = torch.pow(
            self.compute_distance_to_prototypes(samples), self.beta
        )  # [n_task, n_sample, n_class]
        densities = self.compute_densities(
            beta_to_prototypes, n_support
        )  # [n_task, n_sample, n_class]
        responsabilities = (
            self.class_prop.unsqueeze(1)
        ) * densities  # [n_task, n_sample, n_class]
        responsabilities /= responsabilities.sum(2, keepdim=True)

        return responsabilities

    def run_task(self, task_dic, shot):
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

        return logs

    def run_method(self, support, query, y_s, y_q):
        """
        Runs the method on a task

        Args:
            support : torch.Tensor of shape [n_tasks, n_support, feature_dim]
            query : torch.Tensor of shape [n_tasks, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_tasks, n_support]
            y_q : torch.Tensor of shape [n_tasks, n_query]
        """
        t0 = time.time()
        n_tasks, n_support, feature_dim = support.size()
        n_tasks, n_query, _ = query.size()
        n_sample = n_support + n_query

        y_s_one_hot = get_one_hot(y_s, self.n_class)

        self.init_params(support, y_s_one_hot, query)

        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        # Init primal_dual params
        if self.auto_init:
            self.init_hyperparams(all_samples)

        for i in tqdm(range(self.n_iter)):
            old_proto = self.prototypes.clone()
            old_cov = self.cov.clone()
            old_theta = self.theta.clone()
            old_class_prop = self.class_prop.clone()

            # E-step
            # Computes the responsabilities on the query set
            responsabilities_query = self.compute_responsabilities(
                all_samples[:, n_support:], n_support
            )

            # Computing weigths
            weights = torch.zeros(n_tasks, n_support + n_query, self.n_class).to(
                self.device
            )
            weights[:, :n_support] = y_s_one_hot
            weights[:, n_support:] = responsabilities_query

            # M-step
            # Update class proportions
            self.class_prop = weights.sum(1) / n_sample

            # Update distrib parameters
            self.M_step(all_samples, weights)

            t_end = time.time()
            # TODO: Rewrite to get criterions
            criterions = self.get_criterions(
                old_proto, old_cov, old_theta, old_class_prop
            )
            self.record_convergence(timestamp=t_end - t0, criterions=criterions)

        preds_q = self.predict(all_samples[:, n_support:])
        self.record_acc(y_q=y_q, preds_q=preds_q)

    def M_step(self, samples, weights):
        """
        Performs the M-step iterations to update the parameters

        Args:
            samples : torch.Tensor of shape [n_tasks, n_support + n_query, feature_dim]
            weights : torch.Tensor of shape [n_tasks, n_support + n_query, n_class]
            n_support : int
            n_query : int

        Updates :
            self.protoypes : torch.Tensor of shape [n_tasks, n_class, feature_dim]
            self.theta : torch.Tensor of shape [n_tasks, n_sample, n_class]
        """
        # raise NotImplementedError("M-step pas encore compris... c.f code moche a coté")
        n_tasks, n_sample, feature_dim = samples.size()
        dual_u = torch.randn((n_tasks, n_sample, self.n_class, feature_dim)).to(
            self.device
        )
        dual_q = (
            torch.eye(feature_dim)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(n_tasks, self.n_class, 1, 1)
            .to(self.device)
        )
        dual_theta_1 = torch.ones(n_tasks, n_sample, self.n_class).to(self.device)
        dual_theta_2 = torch.ones(n_tasks, n_sample, self.n_class).to(self.device)

        pow_weights = torch.pow(weights, 1 / self.beta)
        crit_list = defaultdict(lambda: [])
        for j in tqdm(range(self.args.n_iter_prox)):
            old_theta, old_proto = self.theta.clone(), self.prototypes.clone()
            old_cov = self.cov.clone()

            q_hat = self.cov - self.gamma * (
                self.zeta1
                * (
                    torch.einsum(
                        "bnci, bncj -> bncij",
                        pow_weights.unsqueeze(-1) * dual_u,
                        samples.unsqueeze(2).repeat(1, 1, self.n_class, 1),
                    ).sum(dim=1, keepdim=False)
                )
                + self.zeta2 * dual_q
            )

            self.cov = self.prox_q(
                (q_hat + q_hat.transpose(-1, -2)) / 2,
                self.class_prop * n_sample * self.gamma,
            )

            self.prototypes = self.prototypes + self.gamma * self.zeta1 * (
                pow_weights.unsqueeze(-1) * dual_u
            ).sum(1)
            new_theta = self.theta - self.gamma * (
                self.zeta3 * dual_theta_1 + self.zeta4 * dual_theta_2
            )
            self.theta = self.prox_theta_1(
                new_theta, (self.kappa - feature_dim * (1 - 1 / self.beta)) * self.gamma
            )

            q_tilde = 2 * self.cov - old_cov
            # assert q_tilde.is_sparse, "matrix not sparse"
            theta_tilde = 2 * self.theta - old_theta
            proto_tilde = 2 * self.prototypes - old_proto

            arg_prox_1 = dual_u + pow_weights.unsqueeze(-1) * (
                torch.einsum("bcij, bnj -> bnci", q_tilde, samples)
                - proto_tilde.unsqueeze(1)
            )
            arg_prox_2 = dual_theta_1 + theta_tilde

            prox_phi = self.prox_phi(
                arg_prox_1, arg_prox_2, (1 / self.zeta1, 1 / self.zeta3)
            )
            dual_u = arg_prox_1 - prox_phi[0]
            dual_theta_1 = arg_prox_2 - prox_phi[1]

            arg_prox_q = dual_q + q_tilde
            dual_q = arg_prox_q - self.prox_theta_2(
                arg_prox_q, self.reg_q * (1 / self.zeta2)
            )

            arg_prox_theta_2 = theta_tilde + dual_theta_2
            dual_theta_2 = arg_prox_theta_2 - self.prox_theta_2(
                arg_prox_theta_2, 1 / ((self.eta**self.p) * self.zeta4)
            )

            crit = (self.theta - old_theta).norm(dim=(1, 2), p=2).mean().cpu().item()
            crit2 = (
                (self.prototypes - old_proto).norm(dim=(1, 2), p=2).mean().cpu().item()
            )
            crit3 = (self.cov - old_cov).norm(dim=(1, 2, 3), p=2).mean().cpu().item()
            crit_list["proto"].append(crit2)
            crit_list["theta"].append(crit)
            crit_list["cov"].append(crit3)

        plt.figure(figsize=(10, 5))
        plt.plot(
            [i for i in range(self.args.n_iter_prox)],
            [c for c in crit_list["theta"]],
            label="theta",
        )
        plt.plot(
            [i for i in range(self.args.n_iter_prox)],
            [c for c in crit_list["proto"]],
            label="proto",
        )
        plt.plot(
            [i for i in range(self.args.n_iter_prox)],
            [c for c in crit_list["cov"]],
            label="cov",
        )

        plt.legend()
        # logscale in y
        plt.yscale("log")
        plt.savefig(f"convergence/{int(torch.rand(1).item()*255)}.png")

        if self.prototypes.isnan().all():
            raise ValueError("Nan values in prototypes")
        # print(torch.min(torch.tensor([c[0] for c in crit_list])), torch.min(torch.tensor([c[1] for c in crit_list])))
        # fig, ax = plt.subplots(figsize=(10, 5))
        # ax.plot([i for i in range(n_iter_prox)], [c[0].cpu() for c in crit_list], label="theta")
        # fig.savefig(f"convergence/theta_{int(torch.rand(1).item()*255)}.png")
        # ax.clear()
        # ax.plot([i for i in range(n_iter_prox)], [c[1].cpu() for c in crit_list], label="proto")
        # fig.savefig(f"convergence/proto_{int(torch.rand(1).item()*255)}.png")
        return

    def prox_theta_1(self, theta, mult):
        """
        Proximal operator for theta

        Args:
            theta : torch.Tensor of shape [n_tasks, n_sample, n_class]
            mult: float (the gamma in prox_{\gamma g})

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class]
        """

        return (theta + torch.sqrt(theta**2 + 4 * mult)) / 2

    def prox_psi_q(self, q, mult):
        """
        Proximal operator for theta

        Args:
            q : torch.Tensor of shape [n_tasks, n_class]
            mult: float (the gamma in prox_{\gamma g})

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class]
        """
        return (q + torch.sqrt(q**2 + 4 * mult.unsqueeze(-1))) / 2

    def prox_q(self, q, mult):
        """
        Proximal operator for -log(det(Q))

        Args:
            q : torch.Tensor of shape [n_tasks, feature_dim, feature_dim]
            mult: float (the gamma in prox_{\gamma g})

        Returns:
            torch.Tensor of shape [n_tasks, feature_dim, feature_dim]
        """
        # eignevalue decomposition of Q
        eigvals, eigvecs = torch.linalg.eigh(q)
        prox = self.prox_psi_q(eigvals, mult)
        return eigvecs @ torch.diag_embed(prox) @ eigvecs.transpose(-1, -2)

    def prox_theta_2(self, theta, mult):
        if self.p == 1:
            return torch.max(
                torch.abs(theta) - mult, torch.zeros_like(theta)
            ) * torch.sign(theta)
        elif self.p == 2:
            return theta / (1 + 2 * mult)
        elif self.p == 3:
            return (
                torch.sign(theta)
                * (torch.sqrt(1 + 12 * mult * torch.abs(theta)) - 1)
                / (6 * mult)
            )
        elif self.p == 4:
            temp = torch.sqrt(theta**2 + 1 / (27 * mult))
            return (
                1
                / (8 * mult) ** (1 / 3)
                * ((theta + temp) ** 1 / 3 - (-theta + temp) ** 1 / 3)
            )

    def prox_phi(self, arg_prox_1, arg_prox_2, mults):
        if len(mults) == 2:
            mult = np.sqrt((mults[0] ** self.beta) / (mults[1] ** (self.beta - 1)))
            u, chi = self.prox_persp(
                arg_prox_1 / np.sqrt(mults[0]),
                arg_prox_2 / np.sqrt(mults[1]),
                mult=mult,
            )
            u = np.sqrt(mults[0]) * u
            chi = np.sqrt(mults[1]) * chi

            return (u, chi)
        elif len(mults) == 1:
            return self.prox_persp(arg_prox_1, arg_prox_2, mult=mults[0])
        else:
            raise ValueError("mults should be of length 1 or 2")

    def prox_persp(self, u, chi, mult):
        """
        Proximal operator for the perspective function

        Args:
            u : torch.Tensor of shape [n_tasks, n_sample, n_class, feature_dim]
            chi : torch.Tensor of shape [n_tasks, n_sample, n_class]
            mult: float (the gamma in prox_{\gamma g})

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class, feature_dim]
        """
        norm_u = torch.norm(u, dim=3)
        beta_star = self.beta / (self.beta - 1)
        rho = (2 * (1 - 1 / beta_star)) ** (beta_star - 1)
        constant = beta_star * (mult ** (beta_star - 1))

        mask_u_zero = (~(u == 0.0)).all(dim=3)
        mask_condition = (constant * chi + rho * torch.pow(norm_u, beta_star)) > 0
        mask = mask_condition & mask_u_zero

        # if we have to solve the persp equation for some inputs
        if mask.sum() != 0:
            shape = chi[mask].shape
            s = np.ones(shape).reshape(-1)  # quite useless to reshape here
            norm_um_np = (
                norm_u[mask]
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
            )
            chi_m_np = (
                chi[mask]
                .cpu()
                .numpy()
                .reshape(
                    -1,
                )
            )
            t_start = time.time()
            s = self.batch_fsolve(s, norm_um_np, chi_m_np, beta_star, rho, mult)
            delta_t = time.time() - t_start
            s = s.reshape(shape)
            sol_persp = torch.tensor(s, dtype=torch.float32).to(self.device)
            u[mask, :], chi[mask] = self.solve_prox_persp(
                u[mask, :], norm_u[mask], chi[mask], mult, sol_persp, beta_star, rho
            )
        u[~mask, :] = 0.0
        chi[(~mask) & (chi <= 0)] = 0.0

        return u, chi

    def persp_equation(self, s, norm_u, chi, beta_star, rho, mult):
        """
        Computes the perspective equation

        Args:
            s : input tensor of shape [n_tasks*n_sample*n_class]
            norm_u : numpy.ndarray of shape [n_tasks*n_sample*n_class]
            chi : numpy.ndarray of shape [n_tasks*n_sample*n_class]
            mult: float (the gamma in prox_{\gamma g})

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class]"""

        first_term = np.power(s, (2 * beta_star - 1))
        second_term = ((beta_star * chi) / (mult * rho)) * np.power(s, beta_star - 1)
        third_term = (beta_star / (rho**2)) * s
        last_term = -1 * (beta_star / (mult * (rho**2))) * norm_u

        return first_term + second_term + third_term + last_term

    def batch_fsolve(self, s, norm_u, chi, beta_star, rho, mult):
        """
        Solves the perspective equation in a batched manner (far more fast than regular fsolve)

        Args:
            s : numpy.ndarray of shape [n_tasks*n_sample*n_class]
            norm_u : numpy.ndarray of shape [n_tasks*n_sample*n_class]
            chi : numpy.ndarray of shape [n_tasks*n_sample*n_class]
            beta_star : float
            rho : float
            mult : float (the gamma in prox_{\gamma g})
        """
        batch_size = self.args.batch_fsolve_size
        size = s.shape[0]
        for i in range(size // batch_size):
            s[i * batch_size : (i + 1) * batch_size] = fsolve(
                self.persp_equation,
                s[i * batch_size : (i + 1) * batch_size],
                args=(
                    norm_u[i * batch_size : (i + 1) * batch_size],
                    chi[i * batch_size : (i + 1) * batch_size],
                    beta_star,
                    rho,
                    mult,
                ),
            )
        return s

    def solve_prox_persp(self, u, norm_u, chi, mult, sol_persp, beta_star, rho):
        """
        Solves the proximal operator for the perspective function

        Args:
            u : torch.Tensor of shape [n_tasks, n_sample, n_class, feature_dim]
            norm_u : torch.Tensor of shape [n_tasks, n_sample, n_class]
            chi : torch.Tensor of shape [n_tasks, n_sample, n_class]
            mult: float (the gamma in prox_{\gamma g})
            sol_persp : torch.Tensor of shape [n_tasks, n_sample, n_class], solution of the equation for the prox
            beta_star : float
            rho : float

        Returns:
            torch.Tensor of shape [n_tasks, n_sample, n_class, feature_dim]
        """
        new_u = u - (mult * (sol_persp.unsqueeze(-1)) * u) / norm_u.unsqueeze(-1)
        new_chi = chi + (mult * rho / beta_star) * torch.pow(sol_persp, beta_star)
        return (new_u, new_chi)

    def get_criterions(self, old_proto, old_cov, old_theta, old_class_prop):
        """
        Computes the criterions

        Args:
            old_proto : torch.Tensor of shape [n_tasks, n_class, feature_dim]
            old_cov : torch.Tensor of shape [n_tasks, feature_dim, feature_dim]
            old_theta : torch.Tensor of shape [n_tasks, n_sample, n_class]
            old_class_prop : torch.Tensor of shape [n_tasks, n_class]

        Returns:
            dict
        """
        with torch.no_grad():
            crit_proto = (self.prototypes - old_proto).norm(dim=[1, 2]).mean().item()
            crit_cov = (self.cov - old_cov).norm(dim=[1, 2, 3], p=2).mean().item()
            crit_theta = (self.theta - old_theta).norm(dim=[1, 2]).mean().item()
            crit_class_prop = (
                (self.class_prop - old_class_prop).norm(dim=1).mean().item()
            )

        return {
            "proto": crit_proto,
            "cov": crit_cov,
            "theta": crit_theta,
            "class_prop": crit_class_prop,
        }
