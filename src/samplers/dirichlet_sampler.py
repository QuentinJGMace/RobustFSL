import torch

from src.samplers.abstract_sampler import AbstractSampler


class Dirichlet_Sampler(AbstractSampler):
    def __init__(self, cat_sampler):
        super(Dirichlet_Sampler, self).__init__(cat_sampler)
        self.name = "Dirichlet_Sampler"
        self.query_class_sampler = torch.distributions.Dirichlet(
            2 * torch.ones(self.k_eff)
        )

    def sample_support(self, classes_support):
        """
        Samples the support set (s-shot samples per class)

        Args:
            classes_support : List of classes for the support set
        Outputs:
            support : List of indexes of the support set
        """
        support = []
        for c in classes_support:
            l = self.m_ind_support[c]
            pos = torch.randperm(l.size(0))
            support.append(l[pos[: self.s_shot]])
        support = torch.cat(support)
        return support

    def sample_query(self, classes_query):
        """
        Samples the query set (q-shot samples per class)

        Args:
            classes_query : List of classes for the query set
        Outputs:
            query : List of indexes of the query set
        """
        query = []
        prop_class = self.query_class_sampler.sample()
        n_per_class = (torch.cumsum(prop_class, 0) * (self.n_query - 1)).floor().int()
        n_per_class[1:] = n_per_class[1:] - n_per_class[:-1]
        n_per_class[-1] = self.n_query - n_per_class[:-1].sum()
        for i, cls in enumerate(classes_query):
            l = self.m_ind_query[cls]
            pos = torch.randperm(l.size(0))
            query.append(l[pos[: n_per_class[i]]])
        query = torch.cat(query)
        return query
