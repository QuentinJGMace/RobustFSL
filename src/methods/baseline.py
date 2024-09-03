import torch.nn.functional as F
from src.methods.abstract_method import AbstractMethod
from src.methods.utils import get_one_hot
from tqdm import tqdm
import torch
import time


class Baseline(AbstractMethod):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone, device, log_file, args)
        self.temp = args.temp
        self.lr = float(args.lr_baseline)
        self.number_tasks = args.batch_size
        self.init_info_lists()
        self.dataset = args.dataset
        self.used_set_support = args.used_set_support

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """

        n_tasks = samples.size(0)
        logits = self.temp * (
            samples.matmul(self.prototypes.transpose(1, 2))
            - 1 / 2 * (self.prototypes**2).sum(2).view(n_tasks, 1, -1)
            - 1 / 2 * (samples**2).sum(2).view(n_tasks, -1, 1)
        )  #
        return logits

    def init_prototypes(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        t0 = time.time()
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s, self.n_class).to(self.device)
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support.to(self.device))
        self.prototypes = weights / counts

        logits_q = self.get_logits(query).detach()
        q_probs = logits_q.softmax(2)
        return q_probs

    def predict(self, query):
        """
        returns:
            preds : torch.Tensor of shape [n_task, n_query]
        """
        logits_q = self.get_logits(query).detach()
        preds_q = logits_q.argmax(2)
        return preds_q

    def record_acc(self, query, y_q):
        """
        inputs:
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_q : torch.Tensor of shape [n_task, q_shot]
        """

        preds_q = self.predict(query)
        self.test_acc.append((preds_q == y_q).float().mean(1, keepdim=True))

    def get_criterions(self, old_prototypes):
        """
        inputs:
            old_prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]

        returns :
            criterions : dict {"ce": float, "mse": float}
        """
        with torch.no_grad():
            crit_prot = (
                torch.norm(self.prototypes - old_prototypes, dim=(-1, -2)).mean().item()
            )
        return {
            "crit_prot": crit_prot,
        }

    def run_method(self, support, query, y_s, y_q):

        """
        Corresponds to the BASELINE inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, q_shot]

        updates :
            self.prototypes : torch.Tensor of shape [n_task, num_class, feature_dim]     # (centroids)
        """
        # Init basic prototypes
        self.init_prototypes(support=support, y_s=y_s, query=query)

        # Record info if there's no Baseline iteration
        if self.n_iter == 0:
            t1 = time.time()
            self.record_acc(query=query, y_q=y_q)
        else:
            self.logger.info(
                " ==> Executing Baseline adaptation over {} iterations on {} shot tasks...".format(
                    self.n_iter, self.args.shots
                )
            )

            self.prototypes.requires_grad_()
            optimizer = torch.optim.Adam([self.prototypes], lr=self.lr)
            y_s_one_hot = get_one_hot(y_s, self.n_class)

            for i in tqdm(range(self.n_iter)):
                old_prototypes = self.prototypes.clone()
                t0 = time.time()

                logits_s = self.get_logits(support)
                ce = (
                    -(y_s_one_hot * torch.log(logits_s.softmax(2) + 1e-12))
                    .sum(2)
                    .mean(1)
                    .sum(0)
                )
                loss = ce

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t1 = time.time()
                criterions = self.get_criterions(old_prototypes)
                self.record_convergence(timestamp=t1 - t0, criterions=criterions)

            self.record_acc(query=query, y_q=y_q)

    def run_task(self, task_dic, shot):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic["y_s"]  # [n_task, shot]
        y_q = task_dic["y_q"]  # [n_task, n_query]
        x_s = task_dic["x_s"]  # [n_task, shot, feature_dim]
        x_q = task_dic["x_q"]  # [n_task, n_query, feature_dim]
        x_mean = task_dic["x_mean"]  # [n_task, feature_dim]

        # Transfer tensors to GPU if needed
        support = x_s.to(self.device)
        query = x_q.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)

        # Perform normalizations
        support, query = self.normalizer(support, query, train_mean=x_mean)

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs
