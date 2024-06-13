import torch
import torchvision.transforms as T

from src.dataset.transform import *
from src.dataset.base_classes import DatasetWrapper


def build_transform(
    size,
    jitter=False,
    enlarge=False,
    augment=False,
):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if enlarge:
        resize = int(size * 256.0 / 224.0)
    else:
        resize = size
    if not augment:
        return T.Compose(
            [
                T.Resize(resize),
                T.CenterCrop(size),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        if jitter:
            return T.Compose(
                [
                    T.RandomResizedCrop(size),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            return T.Compose(
                [
                    T.RandomResizedCrop(size),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )


def build_data_loader(
    data_source=None,
    batch_size=64,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None,
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(data_source, transform=tfm, is_train=is_train),
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()),
    )
    assert len(data_loader) > 0

    return data_loader


def initialize_data_loaders(args, dataset, preprocess):
    """
    Initialisises the data loaders

    Args:
        args : config args
        dataset: The dataset to be used.
        preprocess: Preprocessing function for data.

    Return:
        data_loaders: Data loaders for the dataset.
                    (dict : {"train": train_loader, "val": val_loader, "test": test_loader})
    """
    # if batch size is an argument, use it, otherwise use the default batch size to 1024
    batch_size = args.batch_size if args.batch_size else 1024

    data_loaders = {
        "train": build_data_loader(
            data_source=dataset.train_x,
            batch_size=batch_size,
            is_train=False,
            shuffle=False,
            tfm=preprocess,
        ),
        "val": build_data_loader(
            data_source=dataset.val,
            batch_size=batch_size,
            is_train=False,
            shuffle=False,
            tfm=preprocess,
        ),
        "test": build_data_loader(
            data_source=dataset.test,
            batch_size=batch_size,
            is_train=False,
            shuffle=False,
            tfm=preprocess,
        ),
    }

    return data_loaders
