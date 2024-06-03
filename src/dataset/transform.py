import torch
from PIL import ImageEnhance
import torchvision.transforms as T

transformtypedict = dict(
    Brightness=ImageEnhance.Brightness,
    Contrast=ImageEnhance.Contrast,
    Sharpness=ImageEnhance.Sharpness,
    Color=ImageEnhance.Color,
)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [
            (transformtypedict[k], transformdict[k]) for k in transformdict
        ]

    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha * (randtensor[i] * 2.0 - 1.0) + 1
            out = transformer(out).enhance(r).convert("RGB")

        return out


def build_transform(
    size,
    jitter=False,
    enlarge=False,
    augment=False,
):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if enlarge:
        resize = int(size * 256.0 / 224.0)
    else:
        resize = size
    if not augment:
        return T.Compose(
            [
                T.Resize(resize),
                T.CenterCrop(size),
                T.ToTensor(),
                normalize,
            ]
        )
    else:
        if jitter:
            return T.Compose(
                [
                    T.RandomResizedCrop(size),
                    ImageJitter(dict(Brightness=0.4, Contrast=0.4, Color=0.4)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )
        else:
            return T.Compose(
                [
                    T.RandomResizedCrop(size),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize,
                ]
            )
