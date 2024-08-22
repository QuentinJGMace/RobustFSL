from .mini_imagenet import MiniImageNet
from .imagenet import ImageNet
from .synthetic import SyntheticDataset
from .builders import *
from .normalize_data import *

# Dataset list for FSL tasks
DATASET_LIST = {
    "miniimagenet": MiniImageNet,
    "synthetic": SyntheticDataset,
}

NORMALIZERS = {
    "L2": L2_normalise,
    "train_mean": normalize_by_train_mean,
    "transductive": transductive_normalise,
    "paddle": paddle_normalize,
}
