import os
from collections import OrderedDict
import shutil
import torch

from .resnet import resnet18

backbones_dict = {
    "resnet18": resnet18,
}


def get_backbone(args):
    return backbones_dict[args.backbone](num_classes_train=args.num_classes_train)


def save_checkpoint(
    state, is_best, filename="checkpoint.pth.tar", folder="result/default"
):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    if is_best:
        shutil.copyfile(folder + "/" + filename, folder + "/model_best.pth.tar")


def load_checkpoint(model, model_path, type="best"):
    if type == "best":
        checkpoint = torch.load(
            "{}/model_best.pth.tar".format(model_path),
            map_location=torch.device("cpu"),
        )
    elif type == "last":
        checkpoint = torch.load(
            "{}/checkpoint.pth.tar".format(model_path),
            map_location=torch.device("cpu"),
        )
    else:
        assert False, "type should be in [best, or last], but got {}".format(type)
    state_dict = checkpoint["state_dict"]
    names = []
    for k, v in state_dict.items():
        names.append(k)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:]
                new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
