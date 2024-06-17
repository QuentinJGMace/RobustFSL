import pytest
import torch

from tests.helpers import load_cfg
from src.api.cfg_utils import load_cfg_from_cfg_file, merge_cfg_from_list
from src.methods.rpaddle_gd import MutlNoisePaddle_GD

dummy_samples = torch.tensor(
    [
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ],
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        [
            [1, 2, 3],
            [0, 0, 0],
            [1, 1, 1],
        ],
    ],
    dtype=torch.float32,
)

dummy_query = torch.tensor(
    [
        [
            [4, 4, 4],
            [5, 5, 5],
            [6, 6, 6],
        ],
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ],
        [
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ],
    ],
    dtype=torch.float32,
)

dummy_labels = torch.tensor(
    [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ]
)


def test_MultnoisePaddle():
    """Tests if the MultnoisePaddle class is initialized correctly."""

    args = load_cfg()  # dummy args
    backbone = None  # dummy backbone
    device = "cpu"
    log_file = "log_file"

    method = MutlNoisePaddle_GD(backbone, device, log_file, args)

    assert method.lambd == args.lambd
    assert method.lr == args.lr
    assert method.beta == args.beta
    assert method.kappa == args.kappa
    assert method.id_cov == args.id_cov
    assert method.eta == (2 / method.kappa) ** (1 / 2)
    assert method.timestamps == []
    assert method.criterions == []
    assert method.test_acc == []


def test_init_prototypes():
    """Tests if the prototypes are initialized correctly."""

    args = load_cfg()  # dummy args
    backbone = None  # dummy backbone
    device = "cpu"
    log_file = "log_file"

    method = MutlNoisePaddle_GD(backbone, device, log_file, args)
    method.n_class = 2

    method.init_prototypes(dummy_samples, dummy_labels)

    expected_prototypes = torch.tensor(
        [
            [
                [2, 2, 2],
                [2, 2, 2],
            ],
            [
                [0.5, 0, 0.5],
                [0, 1, 0],
            ],
            [
                [1, 1.5, 2],
                [0, 0, 0],
            ],
        ],
        dtype=torch.float32,
    )

    assert torch.allclose(method.prototypes, expected_prototypes)


def test_compute_malahanobis():
    """Tests if the Mahalanobis distance is computed correctly."""

    args = load_cfg()  # dummy args
    backbone = None  # dummy backbone
    device = "cpu"
    log_file = "log_file"

    method = MutlNoisePaddle_GD(backbone, device, log_file, args)
    method.n_class = 2

    method.init_params(dummy_samples, dummy_labels, dummy_query)

    theta = torch.ones(3, 3)

    mahalanobis = method.compute_mahalanobis(dummy_query, theta)

    assert mahalanobis.size() == (3, 3, 2)
