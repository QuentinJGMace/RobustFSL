import os
from collections import defaultdict

from src.dataset.base_classes import Datum, DatasetBase


class SyntheticDataset(DatasetBase):
    def __init__(self, root):

        self._train_x = None
        self._train_u = None
        self._val = None
        self._test = None

        self._num_classes = 2
        self._classnames = ["class_0", "class_1"]
        self._lab2cname = {0: "class_0", 1: "class_1"}
