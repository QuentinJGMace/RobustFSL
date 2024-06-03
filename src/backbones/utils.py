from .resnet import resnet18

backbones_dict = {
    "resnet18": resnet18,
}


def get_backbone(args):
    return backbones_dict[args.backbone](num_classes_train=args.num_classes_train)
