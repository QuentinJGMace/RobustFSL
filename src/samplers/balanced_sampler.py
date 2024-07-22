import numpy as np
import torch

from src.samplers.abstract_sampler import AbstractSampler


class BalancedSampler(AbstractSampler):
    """
    Sampler that samples a standard support set (of size s_shot * n_class)
    and a query set (of size n_query).

    In the query set, the number of samples per class is fixed and equal to n_query // k_eff.
    """

    def __init__(self, cat_sampler):
        super(BalancedSampler, self).__init__(cat_sampler)
        self.name = "BalancedSampler"
        self.n_query_per_class = self.n_query // self.k_eff

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
        Samples the query set (n_query samples, n_query // k_eff samples per class)

        Args:
            classes_query : List of classes for the query set
        Outputs:
            query : List of indexes of the query set
        """
        query = []
        for c in classes_query:
            l = self.m_ind_query[c]
            pos = torch.randperm(l.size(0))
            query.append(l[pos[: self.n_query_per_class]])
        query = torch.cat(query)
        return query
