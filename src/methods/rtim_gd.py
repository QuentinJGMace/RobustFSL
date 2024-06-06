import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.methods.abstract_method import AbstractMethod
from src.methods.utils import get_one_hot


class RobustTIM(AbstractMethod):
    """
    Transudctive Information Maximisation method with generalised gaussians with gradient descent optimization
    """

    def __init__(self, backbone, device, log_file, args):
        super(RobustTIM, self).__init__(backbone, device, log_file, args)

        self.temp = args.temp
        self.loss_weigths = args.loss_weights.copy()
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support
        self.n_ways = args.n_ways
        self.lr = args.lr

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
        self.mutual_infos = []
        self.entropy = []
        self.cond_entropy = []
        self.test_acc = []
        self.losses = []

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
        self.prototypes = (
            weights / counts
        )  # average of the feature of each classes on the support set

    def record_infos(self, query, y_q):
        """
        Records the logs
        inputs:
            query : torch.Tensor of shape [n_task, query, feature_dim]
            y_q : torch.Tensor of shape [n_task, query]
        """
        logits = self.get_logits(query).detach()
        probs = F.softmax(logits, dim=2)

        self.test_acc.append(self.compute_accuracy(query, y_q))
        self.mutual_infos.append(self.compute_mutual_info(probs))
        self.entropy.append(self.compute_entropy(probs))
        self.cond_entropy.append(self.compute_cond_entropy(probs, y_q))

    def get_logs(self):
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        self.criterions = torch.stack(self.criterions, dim=0).detach().cpu().numpy()
        self.cond_entropy = torch.cat(self.cond_entropy, dim=1).cpu().numpy()
        self.entropy = torch.cat(self.entropy, dim=1).cpu().numpy()
        self.mutual_infos = torch.cat(self.mutual_infos, dim=1).cpu().numpy()
        return {
            "timestamps": self.timestamps,
            "mutual_info": self.mutual_infos,
            "entropy": self.entropy,
            "cond_entropy": self.cond_entropy,
            "acc": self.test_acc,
            "losses": self.losses,
            "criterions": self.criterions,
        }

    def run_task(self, task_dic, shot):
        """
        Fits the model to the FSL tasks
        inputs:
            task_dic : dict    {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        returns:
            logs : dict
        """
        self.logger.info(
            " ==> Executing RobustTIM with lambda = {}, beta = {}".format(
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

        self.init_prototypes(support, y_s)

        # Init optimizer
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)

        # Optimization loop
        for i in range(self.n_iter):

            old_prototypes = self.prototypes
            t_start = time.time()

            # Compute the logits
            logits_support = self.get_logits(support)
            logits_query = self.get_logits(query)

            # Computes loss
            ce = (
                -(y_s_one_hot * torch.log(logits_support.softmax(2) + 1e-12))
                .sum(2)
                .mean(1)
                .sum(0)  # sum on classes, mean on support, sum on tasks
            )

            q_probs = logits_query.softmax(2)
            q_cond_entropy = (
                -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            )
            q_entropy = (
                -(q_probs.mean(1) * torch.log(q_probs.mean(1) + 1e-12)).sum(1).sum(0)
            )

            loss = self.loss_weigths[0] * ce - (
                self.loss_weigths[1] * q_entropy + self.loss_weigths[2] * q_cond_entropy
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # records convergence
            t_end = time.time()
            weight_diff = (old_prototypes - self.prototypes).norm(dim=-1).mean(-1)
            criterions = weight_diff
            self.record_convergence(new_time=t_end - t_start, criterions=criterions)

        self.record_infos(query, y_q)
        logs = self.get_logs()

        return logs


class RobustAlphaTIM(RobustTIM):
    def __init__(self, backbone, device, log_file, args):
        super(RobustAlphaTIM, self).__init__(backbone, device, log_file, args)
        self.lr = float(args.lr_alpha_tim)
        self.entropies = args.entropies.copy()
        self.alpha_values = args.alpha_values

    def run_task(self, task_dic, shot):
        """
        Fits the model to the FSL tasks
        inputs:
            task_dic : dict    {"x_s": torch.Tensor, "y_s": torch.Tensor, "x_q": torch.Tensor, "y_q": torch.Tensor}
            shot : int

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        returns:
            logs : dict
        """
        self.logger.info(
            " ==> Executing RobustTIM with lambda = {}, beta = {}".format(
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

        self.init_prototypes(support, y_s)

        # Init optimizer
        self.prototypes.requires_grad_()
        optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)

        # Optimization loop
        for i in tqdm(range(self.n_iter)):
            old_prototypes = self.prototypes.detach()

            t_start = time.time()

            # Compute the logits
            logits_support = self.get_logits(support)
            logits_query = self.get_logits(query)
            q_probs = logits_query.softmax(2)

            # Computes loss
            if self.entropies[0] == "Shannon":
                ce = (
                    -(y_s_one_hot * torch.log(logits_support.softmax(2) + 1e-12))
                    .sum(2)
                    .mean(1)
                    .sum(0)
                )
            elif self.entropies[0] == "Alpha":
                ce = torch.pow(y_s_one_hot, self.alpha_values[0]) * torch.pow(
                    logits_support.softmax(2) + 1e-12, 1 - self.alpha_values[0]
                )
                ce = ((1 - ce.sum(2)) / (self.alpha_values[0] - 1)).mean(1).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            if self.entropies[1] == "Shannon":
                q_ent = -(q_probs.mean(1) * torch.log(q_probs.mean(1))).sum(1).sum(0)
            elif self.entropies[1] == "Alpha":
                q_ent = (
                    (1 - (torch.pow(q_probs.mean(1), self.alpha_values[1])).sum(1))
                    / (self.alpha_values[1] - 1)
                ).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            if self.entropies[2] == "Shannon":
                q_cond_ent = (
                    -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
                )
            elif self.entropies[2] == "Alpha":
                q_cond_ent = (
                    (
                        (1 - (torch.pow(q_probs + 1e-12, self.alpha_values[2])).sum(2))
                        / (self.alpha_values[2] - 1)
                    )
                    .mean(1)
                    .sum(0)
                )
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Loss computation
            loss = self.loss_weights[0] * ce - (
                self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # records convergence
            t_end = time.time()
            weight_diff = (old_prototypes - self.prototypes).norm(dim=-1).mean(-1)
            criterions = weight_diff
            self.record_convergence(new_time=t_end - t_start, criterions=criterions)

        self.record_infos(query, y_q)
        logs = self.get_logs()

        return logs
