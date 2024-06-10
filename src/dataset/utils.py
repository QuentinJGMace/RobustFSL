from collections import defaultdict

import os
import random
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


def read_image(path):
    """
    Read an image from a file path and return it in the form of a PIL image.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        print(f"Error reading image {path}: {e}")


def read_json(path):
    """
    Read a JSON file from a file path and return it as a dictionary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error reading JSON file {path}: {e}")


def write_json(data, path):
    """
    Write a dictionary to a JSON file.
    """
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=4, separators=(",", ": "))


def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.

    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".") and "sh" not in f]
    if sort:
        items.sort()
    return items


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }
