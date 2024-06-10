import os
import random
import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from src.dataset import build_transform
from src.backbones import get_backbone, load_checkpoint
from src.api.utils import (
    load_cfg_from_cfg_file,
    merge_cfg_from_list,
)
from src.logger.logger import Logger, get_log_file
from src.api.eval_few_shot import Evaluator_few_shot

torch.cuda.empty_cache()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    torch.cuda.set_device(args.device)

    # create model
    preprocess_transform = build_transform(
        size=84, jitter=False, enlarge=args.enlarge, augment=False
    )
    backbone = get_backbone(args).to(device)
    checkpoint_path = args.ckpt_path

    load_checkpoint(backbone, checkpoint_path, device, type="best")
    log_file = get_log_file(
        log_path=args.log_path, dataset=args.dataset, method=args.name_method
    )
    logger = Logger(__name__, log_file)

    if args.shots > 0:  # few-shot setting
        evaluator = Evaluator_few_shot(device=device, args=args, log_file=log_file)
    evaluator.run_full_evaluation(backbone=backbone, preprocess=preprocess_transform)


if __name__ == "__main__":
    main()
