from src.methods.utils import get_one_hot
from src.api.utils import wrap_tqdm
from src.methods.km import KM
from src.dataset import NORMALIZERS
import torch
import time


class Paddle(KM):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)

        if self.args.normalizer == "default":
            self.normalizer = NORMALIZERS["paddle"]

    def __del__(self):
        self.logger.del_logger()

    def A(self, p):
        """
        inputs:
            p : torch.tensor of shape [n_tasks, n_query, num_class]

        returns:
            v : torch.Tensor of shape [n_task, n_query, num_class]
        """

        n_query = p.size(1)
        v = p.sum(1) / n_query
        return v

    def A_adj(self, v, n_query):
        """
        inputs:
            V : torch.tensor of shape [n_tasks, num_class]
            n_query : int

        returns:
            p : torch.Tensor of shape [n_task, n_query, num_class]
        """

        p = v.unsqueeze(1).repeat(1, n_query, 1) / n_query
        return p

    def u_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]

        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """

        n_query = query.size(1)
        logits = self.get_logits(query).detach()
        self.u = (logits + self.lambd * self.A_adj(self.v, n_query)).softmax(2)

    def v_update(self):
        """
        updates:
            self.v : torch.Tensor of shape [n_task, num_class]
        """

        self.v = torch.log(self.A(self.u) + 1e-6) + 1

    def w_update(self, support, query, y_s_one_hot):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, shot, n_ways]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        num = torch.einsum(
            "bkq,bqd->bkd", torch.transpose(self.u, 1, 2), query
        ) + torch.einsum("bkq,bqd->bkd", torch.transpose(y_s_one_hot, 1, 2), support)
        den = self.u.sum(1) + y_s_one_hot.sum(1)
        self.w = torch.div(num, den.unsqueeze(2))

    def run_method(
        self, support, query, y_s, y_q, idx_outliers_support, idx_outliers_query
    ):
        """
        Corresponds to the PADDLE inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]
            idx_outliers_support : torch.Tensor of shape [n_task, n_outliers_support]
            idx_outliers_query : torch.Tensor of shape [n_task, n_outliers_query]

        updates :
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
            self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """

        self.logger.info(" ==> Executing PADDLE with LAMBDA = {}".format(self.lambd))

        y_s_one_hot = get_one_hot(y_s, self.n_class)
        n_task, n_ways = y_s_one_hot.size(0), y_s_one_hot.size(2)

        self.init_w(support=support, y_s=y_s)  # initialize basic prototypes
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)
        self.v = torch.zeros(n_task, n_ways).to(
            self.device
        )  # initialize v to vector of zeros

        for i in wrap_tqdm(range(self.n_iter), disable_tqdm=True):

            w_old = self.w.clone()
            u_old = self.u.clone()
            t0 = time.time()

            self.u_update(query)
            self.v_update()
            self.w_update(support, query, y_s_one_hot)

            t1 = time.time()
            criterions = self.get_criterions(w_old, u_old)
            self.record_convergence(timestamp=t1 - t0, criterions=criterions)

        self.record_acc(y_q=y_q, indexes_outliers_query=idx_outliers_query)
