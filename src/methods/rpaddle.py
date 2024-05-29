import torch
from torch import nn
import torch.nn.functional as F
import time
from tqdm import tqdm

from .abstract_method import AbstractMethod, MinMaxScaler


class RobustPaddle(AbstractMethod):
    """ "Robust PADDLE method for few shot learning"""

    def __init__(self, backbone, device, log_file, args):
        super(RobustPaddle, self).__init__(backbone, device, log_file, args)

        self.lambd = args.lambd
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
        one_hot = self.get_one_hot(y_s)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.prototypes = (
            weights / counts
        )  # average of the feature of each classes on the support set

    def A(self, probs):
        """
        Computes the class ratios estimate in the query set
        inputs:
            probs : torch.Tensor of shape [n_task, n_query, num_class]
        outputs:
            avg : torch.Tensor of shape [n_task, num_class]
        """
        n_query = probs.size(1)
        avg = probs.sum(1) / n_query
        return avg

    def A_adj(self, avg, n_query):
        """
        Dual of A
        inputs:
            avg : torch.Tensor of shape [n_task, num_class]
            n_query : int
        outputs:
            probs : torch.Tensor of shape [n_task, n_query, num_class]
        """
        probs = avg.unsqueeze(1).repeat(1, n_query, 1) / n_query
        return probs

    def update_u(self, query):
        """
        Updates the probabilities estimate in the query set
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]

        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """
        n_query = query.size(1)
        logits = self.get_logits(query).detach()
        self.u = (logits + self.lambd * self.A_adj(self.avg, n_query)).softmax(2)

    def update_v(self):
        """
        Updates dual variable v
        updates:
            self.v : torch.Tensor of shape [n_task, num_class]
        """
        self.avg = torch.log(self.A(self.u) + 1e-6) + 1

    def update_prototypes(self, support, query, y_s_one_hot):
        """
        Updates the prototypes
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, shot, n_ways]

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        num = torch.einsum(
            "bkq,bqd->bkd", torch.transpose(self.u, 1, 2), query
        ) + torch.einsum("bkq,bqd->bkd", torch.transpose(y_s_one_hot, 1, 2), support)
        den = self.u.sum(1) + y_s_one_hot.sum(1)
        self.prototypes = torch.div(num, den.unsqueeze(2))

    def fit(self, task_dic, shot):
        """
        Fits the model
        inputs:
            task_dict : dict    {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot: int
        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        returns:
            logs : dict
        """
        self.logger.info(
            " ==> Executing RobustPADDLE with lambda = {}, beta = {}".format(
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

        n_task, n_ways = y_s_one_hot.size(0), y_s_one_hot.size(2)
        self.init_prototypes(support, y_s)  # initialize prototypes
        self.v = torch.zeros(n_task, n_ways).to(
            self.device
        )  # initialize v to vector of zeros

        # Optimization loop
        for i in tqdm(range(self.n_iter)):

            old_prototypes = self.prototypes
            t_start = time.time()

            self.update_u(query)
            self.update_v()
            self.update_prototypes(support, query, y_s_one_hot)

            # records convergence
            t_end = time.time()
            weight_diff = (old_prototypes - self.prototypes).norm(dim=-1).mean(-1)
            criterions = weight_diff
            self.record_convergence(new_time=t_end - t_start, criterions=criterions)

        self.record_accuracy(acc=self.compute_accuracy(query, self.u))
        logs = self.get_logs()
        return logs
