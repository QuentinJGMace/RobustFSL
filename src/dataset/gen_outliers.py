import torch


def generate_outliers(
    all_features,
    all_labels,
    indices,
    n_outliers=0,
    outlier_type="mult_unif",
    is_support=False,
    save_mult_outlier_distrib=False,
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
    # if no outliers to be added, return the original features and labels
    if (n_outliers) == 0:
        return all_features[indices].clone(), all_labels[indices].clone(), [], None

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
    new_features[mask_outliers], mult = OUTLIER_FUNCTIONS[outlier_type](
        all_features[indices[mask_outliers]]
    )

    new_features[~mask_outliers] = all_features[indices[~mask_outliers]]

    return new_features, new_labels, indices_outliers, mult


def gen_mult_unif_outliers(features: torch.Tensor, scalar_range=(1.0, 5.0)):
    """
    Generate outliers by multiplying the features with a random scalar value.

    Args:
        features : Features to generate outliers from.
        scalar_range : Range of the scalar value to multiply with. (scalar generated from U(low, high))
    """
    low, high = scalar_range
    mult = torch.rand(features.size(0), 1) * (high - low) + low
    mult = mult.to(features.device)
    return features * mult, mult


def gen_mult_randn_outliers(features: torch.Tensor, std=5.0):
    """
    Generate outliers by multiplying the features with a scalar value generated from a normal distribution.

    Args:
        features : Features to generate outliers from.
        mean : Mean of the normal distribution.
        std : Standard deviation of the normal distribution.
    """
    mult = torch.abs(torch.randn(features.size(0), 1)) * std
    mult = mult.to(features.device)
    return features * mult, mult


OUTLIER_FUNCTIONS = {
    "zeros": lambda x: (torch.zeros_like(x), None),
    "randn": lambda x: (torch.randn_like(x), None),
    "rand": lambda x: (torch.rand_like(x), None),
    "mult_unif": gen_mult_unif_outliers,
    "mult_randn": gen_mult_randn_outliers,
}
