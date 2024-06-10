import os
import shutil
import torch.optim
import argparse
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR


def get_scheduler(epochs, num_batches, optimizer, args):

    SCHEDULER = {
        "step": StepLR(optimizer, args.lr_stepsize, args.gamma),
        "multi_step": MultiStepLR(
            optimizer,
            milestones=[int(0.5 * epochs), int(0.75 * epochs)],
            gamma=args.gamma,
        ),
        "cosine": CosineAnnealingLR(optimizer, num_batches * epochs, eta_min=1e-9),
        None: None,
    }
    return SCHEDULER[args.scheduler]


def get_optimizer(
    args: argparse.Namespace, backbone: torch.nn.Module
) -> torch.optim.Optimizer:
    OPTIMIZER = {
        "SGD": torch.optim.SGD(
            backbone.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        ),
        "Adam": torch.optim.Adam(
            backbone.parameters(), lr=args.lr, weight_decay=args.weight_decay
        ),
    }
    return OPTIMIZER[args.optimizer_name]


def save_checkpoint(
    state, is_best, filename="checkpoint.pth.tar", folder="result/default"
):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + "/" + filename, folder + "/model_best.pth.tar")
