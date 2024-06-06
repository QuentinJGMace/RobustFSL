import numpy as np
import torch.backends.cudnn as cudnn
import random
import torch
import wandb
from tqdm import tqdm
from src.dataset import build_transform
from src.backbones import get_backbone
from src.train_backbone.trainer import Trainer
from src.train_backbone.utils import get_optimizer, get_scheduler, save_checkpoint
from src.api.utils import parse_args, init_wandb, wrap_tqdm

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_transform = build_transform(
        size=84, jitter=True, enlarge=False, augment=True
    )

    args = parse_args()

    if args.seed != -1:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    backbone = get_backbone(args).to(device)
    trainer = Trainer(device=device, preprocess=preprocess_transform, args=args)

    optimizer = get_optimizer(args=args, backbone=backbone)
    scheduler = get_scheduler(
        optimizer=optimizer,
        num_batches=len(trainer.train_loader),
        epochs=args.epochs,
        args=args,
    )

    start_epoch, best_prec1 = 0, -1
    init_wandb(args)
    tqdm_loop = wrap_tqdm(list(range(start_epoch, args.epochs)), disable_tqdm=True)

    for epoch in tqdm_loop:
        trainer.do_epoch(
            epoch=epoch,
            scheduler=scheduler,
            print_freq=100,
            disable_tqdm=True,
            callback=None,
            model=backbone,
            alpha=args.alpha,
            optimizer=optimizer,
        )

        # Evaluation on validation set
        prec1 = trainer.meta_val(model=backbone, disable_tqdm=True, epoch=epoch)
        wandb.log({"acc_val": prec1, "epoch": epoch})
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint(
            state={
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": backbone.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            is_best=is_best,
            folder=args.ckpt_path,
        )

        if scheduler is not None:
            scheduler.step()

    wandb.finish()
    print("Training is done!")
