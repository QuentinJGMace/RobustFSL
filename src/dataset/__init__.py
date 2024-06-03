from .utils import build_data_loader
from .mini_imagenet import MiniImageNet
from .imagenet import ImageNet

# Dataset list for FSL tasks
DATASET_LIST = {
    "miniimagenet": MiniImageNet,
}
