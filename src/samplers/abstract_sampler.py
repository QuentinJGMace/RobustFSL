import torch
import numpy as np
from collections import defaultdict

"""This file defines the abstract base to all samplers in the project.
A sampler is an object that yields batches of data (support + query indices) when iterated."""


class CategoriesSampler_few_shot:
    """
    Base class the initialises the list of the indices of data points that belongs to each class

    CategorySampler
        inputs:
            label : All labels of dataset
            n_batch : Number of batches to load
            k_eff : Number of classification ways (k_eff)
            s_shot : Support shot
            n_query : Size of query set
            alpha : Dirichlet's concentration parameter

    """

    def __init__(
        self,
        n_batch,
        k_eff,
        n_class,
        s_shot,
        n_query,
        force_query_size=False,
        n_class_support=None,
    ):
        # the number of iterations in the dataloader
        self.n_batch = n_batch
        self.k_eff = k_eff
        self.s_shot = s_shot
        self.n_query = n_query
        self.n_class = n_class
        if n_class_support is None:
            self.n_class_support = self.n_class
        else:
            self.n_class_support = n_class_support
        assert (
            self.n_class_support <= self.n_class
        ), "n_class_support should be less than n_class"
        assert (
            self.n_class_support >= self.k_eff
        ), "n_class_support should be greater than k_eff"
        self.force_query_size = force_query_size
        # self.list_classes = [i for i in range(n_class)]

    def create_list_classes(self, label_support, label_query):
        """
        Initialise the indexes where each class appears in the support and query sets

        Args :
            label_support : Tensor of labels of the support set
            label_query : Tensor of labels of the query set
        Updates:
            m_ind_support : List of indexes where each class appears in the support set
            m_ind_query : List of indexes where each class appears in the query set
        """
        label_support = np.array(label_support.cpu(), dtype=int)  # all data label
        self.m_ind_support = {}  # the data index of each class
        for i in np.unique(label_support):
            # all data index of this class
            ind = np.argwhere(label_support == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind_support[i] = ind

        label_query = np.array(label_query.cpu())  # all data label
        assert (label_support == label_query).all()
        # print(max(label_support), min(label_support), len(np.unique(label_support)))
        self.m_ind_query = {}  # the data index of each class
        for i in np.unique(label_query):
            # all data index of this class
            ind = np.argwhere(label_query == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind_query[i] = ind

        self.list_classes = list(self.m_ind_support.keys())


class AbstractSampler:
    def __init__(self, cat_sampler):
        self.n_class = cat_sampler.n_class
        self.n_class_support = cat_sampler.n_class_support
        self.list_classes = cat_sampler.list_classes
        self.n_batch = cat_sampler.n_batch
        self.k_eff = cat_sampler.k_eff
        self.s_shot = cat_sampler.s_shot
        self.m_ind_support = cat_sampler.m_ind_support
        self.m_ind_query = cat_sampler.m_ind_query
        self.n_query = cat_sampler.n_query
        self.force_query_size = cat_sampler.force_query_size

    def __len__(self):
        return self.n_batch

    def sample_classes_support_and_query(self):
        """
        Samples n_class_support classes from the list of classes
        and then samples k_eff classes from the n_class_support classes (for the query set)

        Outputs:
            classes_support : List of classes for the support set
            classes_query : List of classes for the query set
        """
        classes_support = [
            self.list_classes[i]
            for i in torch.randperm(len(self.list_classes))[
                : self.n_class_support
            ].tolist()
        ]

        # samples k_eff classes
        classes_query = [
            classes_support[i]
            for i in torch.randperm(len(classes_support))[: self.k_eff].tolist()
        ]

        return classes_support, classes_query

    def intersection(self, indices_support, indices_query):
        """
        Returns True iff there is an intersection between the support and query sets
        Used to ensure there is no intersections between the query and support sets

        Args:
            indices_support : List of indexes of the support set
            indices_query : List of indexes of the query set
        Returns:
            is_intersection : Boolean
        """
        intersection = False
        for i in indices_support:
            if i in indices_query:
                intersection = True
                break
        return intersection

    def sample_support(self, classes_support):
        """
        Samples the support set from the classes in classes_support

        Args:
            classes_support : List of classes for the support set
        Returns:
            support : List of indexes of the support set
        """
        raise NotImplementedError("Should not be used as this is an abstract class")

    def sample_query(self, classes_query):
        """
        Samples the query set from the classes in classes_query

        Args:
            classes_query : List of classes for the query set
        Returns:
            query : List of indexes of the query set
        """
        raise NotImplementedError("Should not be used as this is an abstract class")

    def __iter__(self):
        for _ in range(self.n_batch):
            # samples the classes in the support set
            classes_support, classes_query = self.sample_classes_support_and_query()

            # samples the support set
            support = self.sample_support(classes_support)

            # samples the query set
            counter = 0
            query = []
            while counter == 0 or self.intersection(support, query):
                counter += 1
                query = self.sample_query(classes_query)

            yield support, query
