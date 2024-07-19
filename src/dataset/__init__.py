from .mini_imagenet import MiniImageNet
from .imagenet import ImageNet
from .synthetic import SyntheticDataset
from .builders import *

# Dataset list for FSL tasks
DATASET_LIST = {
    "miniimagenet": MiniImageNet,
    "synthetic": SyntheticDataset,
}
