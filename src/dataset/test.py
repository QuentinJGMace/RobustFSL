import pytest
import os
import random
import torchvision.transforms as T
from .mini_imagenet import MiniImageNet
from .utils import build_data_loader


# tests if the MiniImageNet loads correctly
# parametrize the test with the root path
@pytest.mark.parametrize("root", ["/home/opis/qmace/RobustFSL/data/miniimagenet"])
def test_miniimagenet(root):
    dataset = MiniImageNet(root)
    assert dataset
    assert dataset.train_x
    assert dataset.val
    assert dataset.test

    batch_size = 2
    tfm = T.Compose([T.Resize([244, 244]), T.ToTensor()])
    dataloader_train = build_data_loader(
        data_source=dataset.train_x,
        batch_size=batch_size,
        tfm=tfm,
        is_train=True,
        shuffle=True,
    )

    assert dataloader_train
    counter = 0
    for datum in dataloader_train:
        assert datum
        assert len(datum) == 2

        counter += 1
        if counter > 10:
            break

    dataloader_val = build_data_loader(
        data_source=dataset.val,
        batch_size=batch_size,
        tfm=tfm,
        is_train=True,
        shuffle=True,
    )

    assert dataloader_val
    counter = 0
    for datum in dataloader_val:
        assert datum
        assert len(datum) == 2

        counter += 1
        if counter > 10:
            break

    dataloader_test = build_data_loader(
        data_source=dataset.test,
        batch_size=batch_size,
        tfm=tfm,
        is_train=False,
        shuffle=False,
    )
    assert dataloader_test
    counter = 0
    for datum in dataloader_test:
        assert datum
        assert len(datum) == 2

        counter += 1
        if counter > 10:
            break

    assert (
        len(dataloader_val.dataset)
        + len(dataloader_test.dataset)
        + len(dataloader_train.dataset)
        == 60000
    )
