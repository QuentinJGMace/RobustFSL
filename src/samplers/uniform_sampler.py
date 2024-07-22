import numpy as np
import torch

from src.samplers.abstract_sampler import AbstractSampler


class UniformSampler(AbstractSampler):
    """
    Sampler that samples a standard support set (of size s_shot * n_class)
    and a query set (of size n_query).

    Elements in the query sets are sampled uniformly at random among k_eff classes (also chosen randomly from the n_class classes).
    This is NOT a balanced sampler (i.e the number of samples per class in the query set is not fixed and may vary).
    """

    def __init__(self, cat_sampler):
        super(UniformSampler, self).__init__(cat_sampler)
        self.name = "UniformSampler"

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
        Samples the query set (n_query samples, sampled uniformly at random among the k_eff classes)

        Args:
            classes_query : List of classes for the query set
        Outputs:
            query : List of indexes of the query set
        """
        query = []
        complete_possible_samples = torch.cat(
            [self.m_ind_query[c] for c in classes_query], dim=0
        )
        pos = torch.randperm(complete_possible_samples.size(0))[: self.n_query]
        query = complete_possible_samples[pos]

        return query
