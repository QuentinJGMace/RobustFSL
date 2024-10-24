import torch
from src.api.cfg_utils import CfgNode


def generate_outliers(
    all_features,
    all_labels,
    indices,
    n_outliers=0,
    outlier_params=None,
    is_support=False,
    save_mult_outlier_distrib=False,
    ood_dict=None,
):
    """
    Api level function to randomly generates outliers for the given features and labels.

    Args:
        all_features : Extracted features for the dataset.
        all_labels : Labels for the dataset.
        indices : Indices of the data points that will be taken into the set

    Returns:
        new_features : Features with outliers.
        new_labels : Labels with outliers.
        indices_outliers : Indices of the outliers.
    """
    if outlier_params is None:
        outlier_params = CfgNode(
            {
                "name": "mult_unif",
                "max_outlier_mult": 5,
            }
        )
    # if no outliers to be added, return the original features and labels
    if (n_outliers) == 0:
        return (
            all_features[indices].clone(),
            all_labels[indices].clone(),
            torch.Tensor([]),
            None,
        )

    # Else chooses random indices and apply transformations

    # init new tensors
    new_features = torch.zeros_like(all_features[indices])
    new_labels = all_labels[indices].clone()

    # Select random indices
    perm = torch.randperm(len(indices))
    indices_outliers = indices[perm][:n_outliers]

    # Mask for torch magic to optimize runtime
    mask_outliers = torch.zeros(len(indices), dtype=torch.bool)
    mask_outliers[perm[:n_outliers]] = True

    mult = None

    # generates the outliers
    if outlier_params["name"] != "ood":
        new_features[mask_outliers], mult = OUTLIER_FUNCTIONS[outlier_params["name"]](
            all_features, indices[mask_outliers], outlier_params=outlier_params
        )
    else:
        new_features[mask_outliers], mult = ood_outliers(
            all_features,
            indices[mask_outliers],
            outlier_params=outlier_params,
            ood_dict=ood_dict,
        )

    new_features[~mask_outliers] = all_features[indices[~mask_outliers]]

    return new_features, new_labels, perm[:n_outliers], mult


def gen_mult_unif_outliers(
    features: torch.Tensor, indices: torch.Tensor, outlier_params: CfgNode = None
):
    """
    Generate outliers by multiplying the features with a random scalar value.

    Args:
        features : Features to generate outliers from.
        scalar_range : Range of the scalar value to multiply with. (scalar generated from U(low, high))
    """
    high = outlier_params["max_outlier_mult"]
    low = 1.0
    mult = torch.rand(features[indices].size(0), 1) * (high - low) + low
    mult = mult.to(features.device)
    return features[indices] * mult, mult


def gen_mult_randn_outliers(
    features: torch.Tensor, indices: torch.Tensor, outlier_params: CfgNode = None
):
    """
    Generate outliers by multiplying the features with a scalar value generated from a normal distribution.

    Args:
        features : Features to generate outliers from.
        mean : Mean of the normal distribution.
        std : Standard deviation of the normal distribution.
    """
    std = outlier_params["std_outlier_mult"]
    mult = torch.abs(torch.randn(features[indices].size(0), 1)) * std
    mult = mult.to(features.device)
    return features[indices] * mult, mult


def swap_images(
    features: torch.Tensor, indices: torch.Tensor, outlier_params: CfgNode = None
):
    """
    Generate outliers by swapping the features with another random image.

    Args:
        features : Features to generate outliers from.
        indices : Indices of the features to generate outliers from.
    """
    # Select random indices

    perm = torch.randperm(len(features))
    indices_outliers = perm[: len(indices)]

    return features[indices_outliers], None


def ood_outliers(
    features: torch.Tensor,
    indices: torch.Tensor,
    outlier_params: CfgNode = None,
    ood_dict: dict = None,
):
    """
    Generate outliers by replacing the features with an OOD image.

    Args:
        features : Features to generate outliers from.
        indices : Indices of the features to generate outliers from.
        outlier_params : Parameters for the outlier generation.
        ood_dict : Dictionary containing the OOD features (and labels but we don't care).
    """
    # Select random indices in the ood dataset

    perm = torch.randperm(len(ood_dict["features"]))

    return ood_dict["features"][perm[: len(indices)]].to(features.device), None


def random_disturb_dimensions(
    features: torch.Tensor,
    indices: torch.Tensor,
    outlier_params: CfgNode = None,
):
    """
    Generate outliers by perturbing random dimensions of the features.

    Args:
        features : Features to generate outliers from.
        indices : Indices of the features to generate outliers from.
        outlier_params : Parameters for the outlier generation.
    """
    # Select random indices
    n_features = features.size(1)
    prop_perturbed_features = (
        outlier_params["prop_perturbed_features"]
        if "prop_perturbed_features" in outlier_params
        else 0.3
    )
    mask = torch.rand(len(indices), n_features) < prop_perturbed_features

    perturbation_type = (
        outlier_params["perturbation_type"]
        if "perturbation_type" in outlier_params
        else "randn"
    )

    outliers = torch.clone(features[indices])

    if perturbation_type == "randn":
        std = outlier_params["std"] if "std" in outlier_params else 0.1
        outliers[mask] += std * torch.randn_like(outliers[mask])
    elif perturbation_type == "mult":
        mult = (
            outlier_params["mult_disturb"] if "mult_disturb" in outlier_params else 10
        )
        outliers[mask] *= (mult - 1) * (torch.rand_like(outliers[mask])) + 1
    return outliers, None


OUTLIER_FUNCTIONS = {
    "zeros": lambda x, indices, outlier_params=None: (
        torch.zeros_like(x[indices]),
        None,
    ),
    "randn": lambda x, indices, outlier_params=None: (
        torch.randn_like(x[indices]) * 5,
        None,
    ),
    "rand": lambda x, indices, outlier_params=None: (torch.rand_like(x[indices]), None),
    "mult_unif": gen_mult_unif_outliers,
    "mult_randn": gen_mult_randn_outliers,
    "swap_images": swap_images,
    "disturb_features": random_disturb_dimensions,
    # TODO : multiply 30% of features by a scalar
}
