import os
import random
from collections import defaultdict
import csv
import pickle

import torchvision.transforms as transforms
from src.dataset.utils import read_json
from src.dataset.base_classes import Datum, DatasetBase

dic_classes = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}


class Cifar10(DatasetBase):
    def __init__(self, root):

        self.classes_to_human = dic_classes

        self.image_dir = root

        data = self.read_data(self.image_dir)
        random.seed(0)
        random.shuffle(data)

        classes_train = [0, 2, 4, 6, 8]
        classes_test = [1, 3, 5, 7, 9]

        trainval = [d for d in data if d.label in classes_train]
        test = [d for d in data if d.label in classes_test]

        train = trainval[: int(0.8 * len(trainval))]
        val = trainval[int(0.8 * len(trainval)) :]

        super().__init__(train_x=train, val=val, test=test)

    def read_batch(self, batch_path):
        with open(batch_path, "rb") as f:
            data = pickle.load(f, encoding="bytes")
        return data

    def read_data(self, data_dir):
        data = []
        for classname in dic_classes:
            class_dir = os.path.join(data_dir, classname)
            for impath in os.listdir(class_dir):
                label = dic_classes[classname]
                datum = Datum(
                    impath=os.path.join(data_dir, classname, impath),
                    label=label,
                    classname=classname,
                )
                data.append(datum)
        return data
