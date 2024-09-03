import argparse
import wandb
import numpy as np
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from typing import List

from src.api.cfg_utils import load_cfg_from_cfg_file, merge_cfg_from_list


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


def init_wandb(args):
    # start a new wandb run to track this script
    resume = "allow" if args.resume else "never"
    wandb.init(
        # set the wandb project where this run will be logged
        project="RFSL",
        # track hyperparameters and run metadata
        config=args,
        id=args.name,
        name=args.name,
        resume=resume,
    )


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
