import torch
import pytest

from tests.helpers import load_cfg
from src.api.cfg_utils import load_cfg_from_cfg_file, merge_cfg_from_list
from src.api.sampler_few_shot import SamplerSupportAndQuery, CategoriesSampler_few_shot

support_labels = torch.tensor([0, 1, 2, 3, 0, 2, 3, 1, 4, 3, 2, 1, 0, 4, 4])
query_labels = torch.tensor([0, 1, 2, 3, 0, 2, 3, 1, 4, 3, 2, 1, 0, 4, 4])


def test_init_CatSampler():

    n_batch = 5
    n_class = 5
    n_query = 5
    k_eff = 3
    s_shot = 2
    force_query_size = False

    cat_sampler = CategoriesSampler_few_shot(
        n_batch=n_batch,
        n_class=n_class,
        n_query=n_query,
        k_eff=k_eff,
        s_shot=s_shot,
        force_query_size=force_query_size,
    )

    assert cat_sampler.n_batch == n_batch
    assert cat_sampler.n_class == n_class
    assert cat_sampler.n_query == n_query
    assert cat_sampler.k_eff == k_eff
    assert cat_sampler.s_shot == s_shot
    assert cat_sampler.force_query_size == force_query_size
    assert cat_sampler.list_classes == list(range(n_class))


def test_create_list_classes():
    n_batch = 5
    n_class = 5
    n_query = 5
    k_eff = 3
    s_shot = 2
    force_query_size = False

    cat_sampler = CategoriesSampler_few_shot(
        n_batch=n_batch,
        n_class=n_class,
        n_query=n_query,
        k_eff=k_eff,
        s_shot=s_shot,
        force_query_size=force_query_size,
    )

    cat_sampler.create_list_classes(support_labels, query_labels)

    expected_m_ind_support = torch.tensor(
        [
            [0, 4, 12],
            [1, 7, 11],
            [2, 5, 10],
            [3, 6, 9],
            [8, 13, 14],
        ]
    )

    for i, l in enumerate(cat_sampler.m_ind_support):
        assert (l == expected_m_ind_support[i]).all()
    for i, l in enumerate(cat_sampler.m_ind_query):
        assert (l == expected_m_ind_support[i]).all()


def test_SamplerFewShot_init():
    n_batch = 5
    n_class = 5
    n_query = 5
    k_eff = 3
    s_shot = 2
    force_query_size = False

    cat_sampler = CategoriesSampler_few_shot(
        n_batch=n_batch,
        n_class=n_class,
        n_query=n_query,
        k_eff=k_eff,
        s_shot=s_shot,
        force_query_size=force_query_size,
    )

    cat_sampler.create_list_classes(support_labels, query_labels)

    sampler = SamplerSupportAndQuery(cat_sampler)

    assert sampler.list_classes == list(range(n_class))
    assert sampler.n_batch == n_batch
    assert sampler.s_shot == s_shot


def test_intersection_function():

    torch.manual_seed(0)

    n_batch = 1000
    n_class = 5
    n_query = 2
    k_eff = 3
    s_shot = 1
    force_query_size = False

    cat_sampler = CategoriesSampler_few_shot(
        n_batch=n_batch,
        n_class=n_class,
        n_query=n_query,
        k_eff=k_eff,
        s_shot=s_shot,
        force_query_size=force_query_size,
    )

    cat_sampler.create_list_classes(support_labels, query_labels)

    sampler = SamplerSupportAndQuery(cat_sampler)

    indices1, indices2 = torch.tensor([0, 1, 2, 3, 4]), torch.tensor([5, 6, 7, 8, 9])
    indices3, indices4 = torch.tensor([0, 1, 2, 3, 4]), torch.tensor([5, 6, 2, 8, 9])

    assert not sampler.intersection(indices1, indices2)
    assert sampler.intersection(indices3, indices4)


@pytest.mark.parametrize("seed", [i for i in range(10)])
def test_no_intersection(seed):

    torch.manual_seed(seed)

    n_batch = 1000
    n_class = 5
    n_query = 2
    k_eff = 3
    s_shot = 1
    force_query_size = False

    cat_sampler = CategoriesSampler_few_shot(
        n_batch=n_batch,
        n_class=n_class,
        n_query=n_query,
        k_eff=k_eff,
        s_shot=s_shot,
        force_query_size=force_query_size,
    )

    cat_sampler.create_list_classes(support_labels, query_labels)

    sampler = SamplerSupportAndQuery(cat_sampler)

    for i, (support, query) in enumerate(sampler):
        assert not sampler.intersection(support, query)
