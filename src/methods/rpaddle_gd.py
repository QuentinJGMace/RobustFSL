import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm

from src.methods.abstract_method import AbstractMethod, MinMaxScaler
from src.methods.utils import get_one_hot


class RobustPaddle_GD(AbstractMethod):
    """Robust PADDLE method for few shot learning with gradient descent optimization"""

    def __init__(self, backbone, device, log_file, args):
        super(RobustPaddle_GD, self).__init__(backbone, device, log_file, args)

        self.lambd = args.lambd
        self.lr = args.lr
        self.n_way = args.n_way

        if not hasattr(args, "beta"):
            raise Warning("Beta parameter is not defined, setting it to 2")
        if args.beta <= 1:
            raise ValueError("Beta should be greater than 1")
        self.beta = args.beta

        self.init_info_lists()

    def init_info_lists(self):
        """
        Initializes the lists for logging
        """
        self.timestamps = []
        self.criterions = []
        self.test_acc = []

    def get_logits(self, samples):
        """
        Gets the logits of the samples
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        outputs:
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_tasks = samples.size(0)
        logits = -(1 / 2) * torch.cdist(samples, self.prototypes, p=self.beta)
        return logits

    def init_prototypes(self, support, y_s):
        """
        Initializes the prototypes
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.prototypes = (weights / counts).to(
            self.device
        )  # average of the feature of each classes on the support set

    def fit(self, task_dic, shot):
        """
        Fits the model
        inputs:
            task_dic : dict    {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int
        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        returns:
            logs : dict
        """
        self.logger.info(
            " ==> Executing RobustPADDLE-GD with lambda = {}, beta = {}".format(
                self.lambd, self.beta
            )
        )
        y_s = task_dic["y_s"]  # [n_task, shot]
        y_q = task_dic["y_q"]  # [n_task, n_query]
        support = task_dic["x_s"]  # [n_task, shot, feature_dim]
        query = task_dic["x_q"]  # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Get one hot encoding of the labels
        y_s_one_hot = F.one_hot(y_s, self.n_ways).to(self.device)

        # Perform normalizations
        scaler = MinMaxScaler(feature_range=(0, 1))
        query, support = scaler(query, support)

        # Inits the prototypes and u
        self.init_prototypes(support, y_s)
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)
        self.prototypes.requires_grad_()
        self.u.requires_grad_()

        # Init logs
        self.init_info_lists()

        # Initialize the optimizer
        optimizer = torch.optim.Adam([self.prototypes, self.u], lr=self.lr)

        # Concatenate support and query
        all_samples = torch.cat([support, query], 1)

        for i in tqdm(range(self.n_iter)):

            old_prototypes = self.prototypes.detach()
            tstart = time.time()

            # data fitting term
            distances = torch.cdist(all_samples, self.prototypes) ** self.beta
            complete_u = torch.cat([y_s_one_hot, self.u], dim=1)

            data_fitting = 1 / 2 * (distances * complete_u).sum((-2, -1)).sum(0)

            # Regularization term (partition complexity)
            query_class_ratios = self.u.mean(1).to(self.device)
            partition_complexity = (
                (query_class_ratios * torch.log(query_class_ratios + 1e-12))
                .sum(-1)
                .sum(0)
            )

            # Loss computation
            loss = (data_fitting - self.lambd * partition_complexity).to(self.device)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Projection into simplex
            with torch.no_grad():
                self.u = self.simplex_project(self.u)
                weight_diff = (old_prototypes - self.prototypes).norm(dim=-1).mean(-1)
                criterions = weight_diff

            t_end = time.time()
            self.record_convergence(timestamp=t_end - tstart, criterions=criterions)

        self.record_accuracy(acc=self.compute_accuracy(y_q, self.u))
        logs = self.get_logs()
        return logs

    def simplex_project(self, u: torch.Tensor, l=1.0):
        """
        Taken from https://www.researchgate.net/publication/283568278_NumPy_SciPy_Recipes_for_Data_Science_Computing_Nearest_Neighbors
        u: [n_tasks, n_q, K]
        """

        # Put in the right form for the function
        matX = u.permute(0, 2, 1).detach().cpu().numpy()

        # Core function
        n_tasks, m, n = matX.shape
        matS = -np.sort(-matX, axis=1)
        matC = np.cumsum(matS, axis=1) - l
        matH = matS - matC / (np.arange(m) + 1).reshape(1, m, 1)
        matH[matH <= 0] = np.inf

        r = np.argmin(matH, axis=1)
        t = []
        for task in range(n_tasks):
            t.append(matC[task, r[task], np.arange(n)] / (r[task] + 1))
        t = np.stack(t, 0)
        matY = matX - t[:, None, :]
        matY[matY < 0] = 0

        # Back to torch
        matY = torch.from_numpy(matY).permute(0, 2, 1).to(self.device)

        assert torch.allclose(
            matY.sum(-1), torch.ones_like(matY.sum(-1))
        ), "Simplex constraint does not seem satisfied"

        return matY


class MultNoisePaddle(RobustPaddle_GD):
    def __init__(self, backbone, device, log_file, args):
        super(RobustPaddle_GD, self).__init__(backbone, device, log_file, args)

        self.lambd = args.lambd
        self.lr = args.lr
        self.n_way = args.n_way

        if not hasattr(args, "beta"):
            raise Warning("Beta parameter is not defined, setting it to 2")
        if args.beta <= 1:
            raise ValueError("Beta should be greater than 1")
        self.beta = args.beta

        self.kappa = args.kappa
        self.eta = (2 / self.kappa) ** (1 / 2)

        self.init_info_lists()

    def get_logits(self, samples):
        """
        Gets the logits of the samples
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        outputs:
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_tasks = samples.size(0)
        logits = (
            -(1 / 2)
            * torch.cdist(samples, self.prototypes, p=self.beta)
            / (self.theta ** (self.beta - 1))
        )
        return logits

    def init_params(self, support, y_s, n_sample):
        """
        Initializes the prototypes
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            n_sample : number of samples in support + query
        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.prototypes = (weights / counts).to(
            self.device
        )  # average of the feature of each classes on the support set
        self.theta = torch.ones(n_tasks, n_sample).to(self.device)

    def fit(self, task_dic, shot):
        """
        Fits the model
        inputs:
            task_dic : dict    {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int
        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        returns:
            logs : dict
        """
        self.logger.info(
            " ==> Executing RobustPADDLE-GD with lambda = {}, beta = {}".format(
                self.lambd, self.beta
            )
        )
        y_s = task_dic["y_s"]  # [n_task, shot]
        y_q = task_dic["y_q"]  # [n_task, n_query]
        support = task_dic["x_s"]  # [n_task, shot, feature_dim]
        query = task_dic["x_q"]  # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Get one hot encoding of the labels
        y_s_one_hot = F.one_hot(y_s, self.n_ways).to(self.device)

        # Perform normalizations
        scaler = MinMaxScaler(feature_range=(0, 1))
        query, support = scaler(query, support)

        # Inits the prototypes and u
        self.init_prototypes(support, y_s)
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)
        self.prototypes.requires_grad_()
        self.u.requires_grad_()

        # Init logs
        self.init_info_lists()

        # Initialize the optimizer
        optimizer = torch.optim.Adam([self.prototypes, self.u], lr=self.lr)

        # Concatenate support and query
        all_samples = torch.cat([support, query], 1)

        for i in tqdm(range(self.n_iter)):

            old_prototypes = self.prototypes.detach()
            tstart = time.time()

            # data fitting term
            distances = torch.cdist(all_samples, self.prototypes) ** self.beta
            distances = distances / (self.theta ** (self.beta - 1))
            complete_u = torch.cat([y_s_one_hot, self.u], dim=1)

            data_fitting = 1 / 2 * (distances * complete_u).sum((-2, -1)).sum(0)

            # Term for theta
            sample_dim = all_samples.size(-1)
            theta_term = (sample_dim * (1 - 1 / self.beta) - self.kappa) * nn.log(
                self.theta
            ).sum(1).sum(0) + (1 / self.eta) * torch.norm(self.theta, dim=1, p=2).sum(0)

            # Regularization term (partition complexity)
            query_class_ratios = self.u.mean(1).to(self.device)
            partition_complexity = (
                (query_class_ratios * torch.log(query_class_ratios + 1e-12))
                .sum(-1)
                .sum(0)
            )

            # Loss computation
            loss = (data_fitting - self.lambd * partition_complexity + theta_term).to(
                self.device
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Projection into simplex
            with torch.no_grad():
                self.u = self.simplex_project(self.u)
                weight_diff = (old_prototypes - self.prototypes).norm(dim=-1).mean(-1)
                criterions = weight_diff

            t_end = time.time()
            self.record_convergence(timestamp=t_end - tstart, criterions=criterions)

        self.record_accuracy(acc=self.compute_accuracy(y_q, self.u))
        logs = self.get_logs()
        return logs
