import argparse

# import wandb
import numpy as np
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from typing import List

from src.api.cfg_utils import load_cfg_from_cfg_file, merge_cfg_from_list
import copy


def save_pickle(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def wrap_tqdm(data_loader, disable_tqdm):
    if disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm(data_loader, total=len(data_loader))
    return tqdm_loader


def parse_args(method=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main")
    cfg = load_cfg_from_cfg_file("config/main_config.yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    dataset_config = "config/datasets_config/config_{}.yaml".format(cfg.dataset)
    if method:
        method_config = "config/methods_config/{}.yaml".format(method)
    else:
        method_config = "config/methods_config/{}.yaml".format(cfg.method)
    backbone_config = "config/backbones_config/{}.yaml".format(cfg.backbone)
    cfg.update(load_cfg_from_cfg_file(dataset_config))
    cfg.update(load_cfg_from_cfg_file(method_config))
    cfg.update(load_cfg_from_cfg_file(backbone_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    cfg.n_class = cfg.num_classes_test
    return cfg


def load_main_config():

    cfg = load_cfg_from_cfg_file("config/main_config.yaml")

    return cfg


def merge_method_cfg(cfg, method_name):
    """Merges a method in the current config

    Does not modify the old config, returns a new one
    """
    method_config_path = "config/methods_config/{}.yaml".format(method_name)
    modified_cfg = copy.deepcopy(cfg)

    modified_cfg.update(load_cfg_from_cfg_file(method_config_path))
    return modified_cfg


def merge_backbone_cfg(cfg, backbone_name):

    """Merges a backbone in the current config

    Does not modify the old config, returns a new one"""

    backbone_config_path = "config/backbones_config/{}.yaml".format(backbone_name)
    modified_cfg = copy.deepcopy(cfg)

    modified_cfg.update(load_cfg_from_cfg_file(backbone_config_path))
    return modified_cfg


def merge_dataset_cfg(cfg, dataset_name):

    """Merges a dataset in the current config

    Does not modify the old config, returns a new one"""

    dataset_config_path = "config/datasets_config/config_{}.yaml".format(dataset_name)
    modified_cfg = copy.deepcopy(cfg)

    modified_cfg.update(load_cfg_from_cfg_file(dataset_config_path))
    return modified_cfg
