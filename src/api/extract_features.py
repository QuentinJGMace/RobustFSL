import os
import random
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as T
from tqdm import tqdm

from src.backbones import get_backbone
from src.dataset import DATASET_LIST, build_transform, initialize_data_loaders
from src.api.utils import (
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
    Logger,
    get_log_file,
    save_pickle,
)

torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def extract_features(args, backbone, data_loader, set_name, device):
    """
    Extracts features for the given data loader and saves them.

    Args:
        args : experiment config
        backbone: The model to be evaluated.
        data_loader: Data loader for the dataset.
        set_name: Name of the set to be saved.
    Returns:
        None (saves the features on disk)
    """
    features = []
    labels = []

    all_features, all_labels = None, None

    for i, (data, target) in enumerate(tqdm(data_loader)):
        data = data.to(device)
        with torch.no_grad():
            features = backbone(data)
            features /= features.norm(dim=-1, keepdim=True)

            if i == 0:
                all_features = features
                all_labels = target.cpu()
            else:
                all_features = torch.cat((all_features, features), dim=0)
                all_labels = torch.cat((all_labels, target.cpu()), dim=0)

    try:
        os.mkdir(f"data/{args.dataset}/saved_features/")
    except:
        pass

    filepath = os.path.join(
        f"data/{args.dataset}/saved_features/{set_name}_features_{args.backbone}.pkl"
    )
    save_pickle(
        filepath,
        {
            "features": all_features,
            "labels": all_labels,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main")
    cfg = load_cfg_from_cfg_file("config/main_config.yaml")
    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    dataset_config = "config/datasets_config/config_{}.yaml".format(cfg.dataset)
    method_config = "config/methods_config/{}.yaml".format(cfg.method)
    backbone_config = "config/backbones_config/{}.yaml".format(cfg.backbone)
    cfg.update(load_cfg_from_cfg_file(dataset_config))
    cfg.update(load_cfg_from_cfg_file(method_config))
    cfg.update(load_cfg_from_cfg_file(backbone_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    cfg.n_class = cfg.num_classes_test

    return cfg


def main():
    args = parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    dataset = DATASET_LIST[args.dataset](args.dataset_path)
    preprocess_transform = build_transform(
        size=84, jitter=False, enlarge=args.enlarge, augment=False
    )
    data_loaders = initialize_data_loaders(
        args=args,
        dataset=dataset,
        preprocess=preprocess_transform,
    )
    backbone = get_backbone(args).to(device)

    if not os.path.exists(
        f"data/{args.dataset}/saved_features/train_features_{args.backbone}.pkl"
    ):
        print("Extracting features from train set")
        extract_features(backbone, data_loaders["train"], "train", device=device)
    if not os.path.exists(
        f"data/{args.dataset}/saved_features/val_features_{args.backbone}.pkl"
    ):
        print("Extracting features from val set")
        extract_features(backbone, data_loaders["val"], "val", device=device)
    if not os.path.exists(
        f"data/{args.dataset}/saved_features/test_features_{args.backbone}.pkl"
    ):
        print("Extracting features from test set")
        extract_features(backbone, data_loaders["test"], "test", device=device)
