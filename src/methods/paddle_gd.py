import torch.nn.functional as F
from tqdm import tqdm
import torch
import time
from src.methods.km import KM
from src.methods.utils import simplex_project


class Paddle_GD(KM):
    def __init__(self, backbone, device, log_file, args):
        super().__init__(backbone=backbone, device=device, log_file=log_file, args=args)
        self.lambd = args.lambd
        self.lr = args.lr
        self.init_info_lists()

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the PADDLE-GD inference (ablation)
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.u : torch.Tensor of shape [n_task, n_query]
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        self.logger.info(" ==> Executing PADDLE GD with LAMBDA = {}".format(self.lambd))

        t0 = time.time()
        y_s_one_hot = F.one_hot(y_s, self.n_class).to(self.device)

        self.init_w(support, y_s)
        # Initialize the soft labels u and the prototypes w
        self.u = (self.get_logits(query)).softmax(-1).to(self.device)
        self.u.requires_grad_()
        self.w.requires_grad_()
        optimizer = torch.optim.Adam([self.w, self.u], lr=self.lr)

        all_samples = torch.cat([support.to(self.device), query.to(self.device)], 1)

        for i in tqdm(range(self.n_iter)):

            w_old = self.w.detach()
            t0 = time.time()

            # Data fitting term
            l2_distances = (
                torch.cdist(all_samples, self.w) ** 2
            )  # [n_tasks, n_query + shot, K]
            all_p = torch.cat(
                [y_s_one_hot.float(), self.u.float()], dim=1
            )  # [n_task s, n_query + shot, K]
            data_fitting = 1 / 2 * (l2_distances * all_p).sum((-2, -1)).sum(0)

            # Complexity term
            marg_u = self.u.mean(1).to(self.device)  # [n_tasks, num_classes]
            marg_ent = -(marg_u * torch.log(marg_u + 1e-12)).sum(-1).sum(0)  # [n_tasks]
            loss = (data_fitting - self.lambd * marg_ent).to(self.device)

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Projection
            with torch.no_grad():
                self.u = simplex_project(self.u, device=self.device)
                weight_diff = (w_old - self.w).norm(dim=-1).mean(-1)
                criterions = weight_diff

            t1 = time.time()
            self.record_convergence(timestamp=t1 - t0, criterions=criterions)

        self.record_acc(y_q=y_q)
