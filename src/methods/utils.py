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


def simplex_project(u: torch.Tensor, device, l=1.0):
    """
    Taken from https://www.researchgate.net/publication/283568278_NumPy_SciPy_Recipes_for_Data_Science_Computing_Nearest_Neighbors
    u: [n_tasks, n_q, K]
    """

    # Put in the right form for the function
    matX = u.permute(0, 2, 1).detach().cpu().numpy()

    # Core function
    n_tasks, m, n = matX.shape
    matS = -np.sort(-matX, axis=1)
    matC = np.cumsum(matS, axis=1) - l
    matH = matS - matC / (np.arange(m) + 1).reshape(1, m, 1)
    matH[matH <= 0] = np.inf

    r = np.argmin(matH, axis=1)
    t = []
    for task in range(n_tasks):
        t.append(matC[task, r[task], np.arange(n)] / (r[task] + 1))
    t = np.stack(t, 0)
    matY = matX - t[:, None, :]
    matY[matY < 0] = 0

    # Back to torch
    matY = torch.from_numpy(matY).permute(0, 2, 1).to(device)

    assert torch.allclose(
        matY.sum(-1), torch.ones_like(matY.sum(-1))
    ), "Simplex constraint does not seem satisfied"

    return matY
