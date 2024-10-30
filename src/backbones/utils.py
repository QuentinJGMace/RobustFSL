import os
from collections import OrderedDict
import shutil
import torch

from .resnet import resnet18
from .feat_resnet import feat_resnet12

backbones_dict = {
    "resnet18": resnet18,
    "feat_resnet12": feat_resnet12,
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


def load_checkpoint(model, model_path, device, type="best"):
    if type == "best":
        if os.path.exists("{}/model_best.pth.tar".format(model_path)):
            checkpoint = torch.load(
                "{}/model_best.pth.tar".format(model_path),
                map_location=torch.device(device),
            )
        elif os.path.exists("{}/model_best.pth".format(model_path)):
            checkpoint = torch.load(
                "{}/model_best.pth".format(model_path),
                map_location=torch.device(device),
            )
    elif type == "last":
        checkpoint = torch.load(
            "{}/checkpoint.pth.tar".format(model_path),
            map_location=torch.device(device),
        )
    else:
        assert False, "type should be in [best, or last], but got {}".format(type)
    try:
        state_dict = checkpoint["state_dict"]
    except KeyError:
        state_dict = checkpoint["params"]
    names = []
    for k, v in state_dict.items():
        names.append(k)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_state_dict = OrderedDict()
        has_to_be_strict = True
        for k, v in state_dict.items():
            if k.startswith("module."):
                name = k[7:]
                if name.startswith("fc."):
                    continue
                new_state_dict[name] = v
            elif k.startswith("encoder."):
                name = k[8:]
                if name.startswith("fc."):
                    continue
                new_state_dict[name] = v

                # has_to_be_strict = False # no fully connected layer in this checkpoint
        # print(model.state_dict().keys())
        # print("--------------------")
        # print(new_state_dict.keys())
        model.load_state_dict(new_state_dict, strict=False)
