import torch
from src.dataset.base_classes import DatasetWrapper


def build_data_loader(
    data_source=None,
    batch_size=64,
    input_size=224,
    tfm=None,
    is_train=True,
    shuffle=False,
    dataset_wrapper=None,
):

    if dataset_wrapper is None:
        dataset_wrapper = DatasetWrapper

    # Build data loader
    data_loader = torch.utils.data.DataLoader(
        dataset_wrapper(
            data_source, input_size=input_size, transform=tfm, is_train=is_train
        ),
        batch_size=batch_size,
        num_workers=0,
        shuffle=shuffle,
        drop_last=False,
        pin_memory=(torch.cuda.is_available()),
    )
    print(data_loader)
    assert len(data_loader) > 0

    return data_loader


def initialize_data_loaders(self, dataset, preprocess):
    """
    Initialisises the data loaders

    Args:
        dataset: The dataset to be used.
        preprocess: Preprocessing function for data.

    Return:
        data_loaders: Data loaders for the dataset.
                    (dict : {"train": train_loader, "val": val_loader, "test": test_loader})
    """
    # if batch size is an argument, use it, otherwise use the default batch size to 1024
    batch_size = self.args.batch_size if self.args.batch_size else 1024

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
