import pytest
import torch

from tests.helpers import load_cfg
from src.api.cfg_utils import load_cfg_from_cfg_file, merge_cfg_from_list
from src.methods.abstract_method import AbstractMethod, MinMaxScaler


def load_cfg():
    cfg = load_cfg_from_cfg_file("config/main_config.yaml")
    dataset_config = "config/datasets_config/config_{}.yaml".format(cfg.dataset)
    method_config = "config/methods_config/{}.yaml".format(cfg.method)
    backbone_config = "config/backbones_config/{}.yaml".format(cfg.backbone)
    cfg.update(load_cfg_from_cfg_file(dataset_config))
    cfg.update(load_cfg_from_cfg_file(method_config))
    cfg.update(load_cfg_from_cfg_file(backbone_config))

    cfg.n_class = cfg.num_classes_test

    return cfg


def test_abstract_init():
    """Tests if the Abstract method class initialises correctly"""
    args = load_cfg()  # dummy args
    backbone = None  # dummy backbone
    device = torch.device("cpu")
    log_file = "test.log"
    method = AbstractMethod(
        backbone=backbone, device=device, log_file=log_file, args=args
    )

    assert method.device == device
    assert method.n_iter == args.iter
    assert method.n_class == args.n_class
    assert method.backbone == backbone
    assert method.log_file == log_file
    assert method.args == args


def test_record_acc():
    """
    Test if the record_acc method works correctly
    """
    args = load_cfg()  # dummy args
    backbone = None  # dummy backbone
    device = torch.device("cpu")
    log_file = "test.log"
    method = AbstractMethod(
        backbone=backbone, device=device, log_file=log_file, args=args
    )
    method.init_info_lists()
    method.predict = lambda: torch.tensor(
        [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ]
    )
    y_q = torch.tensor([[0, 1, 0], [1, 0, 1], [1, 0, 0]])
    method.record_acc(y_q)

    assert len(method.test_acc) == 1
    assert method.test_acc[0].shape == (3, 1)
    assert torch.allclose(method.test_acc[0], torch.tensor([[1 / 3], [0.0], [1.0]]))
