import os
import shutil
from huggingface_hub import hf_hub_download


def load_backbone_from_hub(model_name: str, savepath: str):
    """Loads a backbone from the huggingface hub and saves it in the savepath"""

    if model_name == "resnet18":
        model_url = "QuentinJG/ResNet18-miniimagenet"
        model_path = hf_hub_download(model_url, filename="model_best.pth.tar")
    elif model_name == "feat_resnet12":
        model_url = "QuentinJG/FeatResNet12-miniimagenet"
        model_path = hf_hub_download(model_url, filename="model_best.pth")

    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    # saves the model in the save_path
    shutil.move(model_path, savepath)


if __name__ == "__main__":
    print("Loading ResNet18 backbone from the huggingface hub")
    load_backbone_from_hub(
        "resnet18", "checkpoints/mini/softmax/resnet18/model_best.pth.tar"
    )

    print("Loading FeatResNet12 backbone from the huggingface hub")
    load_backbone_from_hub(
        "feat_resnet12", "checkpoints/mini/softmax/feat_resnet12/model_best.pth"
    )
