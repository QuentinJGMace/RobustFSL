from .mini_imagenet import MiniImageNet
from .imagenet import ImageNet
from .builders import *

# Dataset list for FSL tasks
DATASET_LIST = {
    "miniimagenet": MiniImageNet,
}
