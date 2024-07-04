import torch
import numpy as np
import shutil
from tqdm import tqdm
import logging
import os
import pickle
import torch.nn.functional as F
from typing import List
import yaml
from ast import literal_eval
import logging
import copy


def get_one_hot(y_s, n_class):

    eye = torch.eye(n_class).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, dim=0)
    return one_hot


def simplex_project(tensor, device):
    """Project a tensor onto the simplex.

    Args:
        tensor (torch.Tensor): Input tensor of shape (..., n) where n is the size of the last dimension.

    Returns:
        torch.Tensor: The projected tensor of the same shape as the input.
    """
    # Get the shape of the input tensor
    shape = tensor.shape
    # Reshape the tensor to 2D: (batch_size, n)
    tensor_reshaped = tensor.view(-1, shape[-1])

    # Sort the tensor along the last dimension in descending order
    sorted_tensor, _ = torch.sort(tensor_reshaped, descending=True, dim=1)

    # Compute the cumulative sum of the sorted tensor along the last dimension
    cumsum_tensor = torch.cumsum(sorted_tensor, dim=1)

    # Create a vector [1/k, ..., 1] for the computation of the threshold
    k = (
        torch.arange(
            1, tensor_reshaped.size(1) + 1, dtype=tensor.dtype, device=tensor.device
        )
        .view(1, -1)
        .to(device)
    )

    # Compute the threshold t
    t = (cumsum_tensor - 1) / k

    # Find the last index where sorted_tensor > t
    mask = sorted_tensor > t
    rho = torch.sum(mask, dim=1) - 1

    # Gather the threshold theta for each sample in the batch
    theta = t[torch.arange(t.size(0)), rho]

    # Perform the projection
    projected_tensor = torch.clamp(tensor_reshaped - theta.view(-1, 1), min=0)

    # Reshape the projected tensor back to the original shape
    projected_tensor = projected_tensor.view(*shape)

    # assert torch.allclose(
    #     projected_tensor.sum(-1), torch.ones_like(projected_tensor.sum(-1))
    # ), "Simplex constraint does not seem satisfied"

    if not torch.allclose(
        projected_tensor.sum(-1), torch.ones_like(projected_tensor.sum(-1))
    ):
        print("Simplex constraint does not seem satisfied")
        print(tensor)
        raise ValueError("Simplex constraint does not seem satisfied")
    return projected_tensor
