import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def get_one_hot(y_s, n_class):

    eye = torch.eye(n_class).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, dim=0)
    return one_hot


def empirical_cov(support, labels, n_class, device):
    """
    Compute the empirical covariance matrix for each class using the corresponding subset of the support set

    Args:
        support: Tensor of shape (batch_size, n_support, feature_dim)
        labels: Tensor of shape (batch_size, n_support)
        device: torch.device
    """
    batch_size, n_support, feature_dim = support.size()

    # Compute the empirical covariance matrix for each class
    cov = torch.zeros(batch_size, n_class, feature_dim, feature_dim).to(device)

    one_hot_support = F.one_hot(labels, n_class).to(device).float()
    class_counts = one_hot_support.sum(dim=1)

    class_means = (one_hot_support.transpose(1, 2) @ support) / class_counts.unsqueeze(
        -1
    )
    centered_features = support.unsqueeze(2) - class_means.unsqueeze(1)
    mask = one_hot_support.unsqueeze(-1)
    centered_features = (
        centered_features * mask
    )  # zero out features that are not in the class
    cov = torch.einsum("bnci, bncj->bncij", centered_features, centered_features).sum(
        dim=1
    ) / (class_counts.unsqueeze(-1).unsqueeze(-1) - 1)

    return cov


def prox_g(S, gamma):  # g(S) = -logdet(S)
    s, U = torch.linalg.eigh(S)
    d = (s + torch.sqrt(s**2 + 4 * gamma)) / 2
    P = (U * d.unsqueeze(-2)).matmul(U.transpose(-1, -2))
    return P


def prox_f(S, gamma):  # f(S) = lambda*sum_{i!=j}|Sij|
    m = torch.diag(
        S.new_ones(S.size(-1), dtype=bool)
    )  # True sur diagonale, False ailleurs
    m_bar = ~m  # False sur diagonale, True ailleurs
    S[..., m_bar] = F.softshrink(
        S[..., m_bar], lambd=gamma
    )  # softshrink(x)=x-lambda if x-lambda>0, x+lambda if x+lambda>0, and 0 otherwise
    S[..., m] = F.softshrink(S[..., m], lambd=0)  # useless i guess ?
    return S


def GLASSO(C, S_0, lambd, max_iter=20000, eps=5e-3):
    """
    inputs:
        C : torch.Tensor of shape [batch_size, n_class, feature_dim, feature_dim]
        S_0 : torch.Tensor of shape [batch_size, n_class, feature_dim, feature_dim]
        lambd : scalar
    """

    y_n = x_n = S_0.clone()
    n_task, n_class, feature_dim = S_0.size(0), S_0.size(1), S_0.size(-1)
    gamma = 1
    lambda_n = 1
    criterion = 1
    i = 0
    for i in range(max_iter):
        x_n_plus_1 = prox_g(
            y_n - gamma * C, gamma
        )  # prox_gamma * g(y_n - gamma*C) = prox_{ gamma*g(.) + <C|.> } (y_n)

        x_n = x_n_plus_1.detach().clone()
        Z = prox_f(2 * x_n - y_n, gamma * lambd)
        y_n = y_n + lambda_n * (Z - x_n)
        criterion = torch.norm(Z - x_n) * 2 / (n_task * n_class * feature_dim * 2)
        if criterion < eps:
            break

    if i >= max_iter:
        print("GLASSO stopped after max number of iterations")
    else:
        print("GLASSO completed in ", i, "iterations")
    # return x_n
    return Z


def Glasso_cov_estimate(support, labels, n_class, device):
    """
    Estimate the covariance matrix using the Glasso method

    Args:
        support: Tensor of shape (batch_size, n_support, feature_dim)
        labels: Tensor of shape (batch_size, n_support)
        device: torch.device
    """

    emp_cov = empirical_cov(support, labels, n_class, device)
    plt.figure()
    plt.imshow(emp_cov[0, 0].cpu().numpy())
    plt.colorbar()
    plt.savefig("emp_cov.png")
    print(emp_cov[0, 0].diagonal())
    S_0 = (
        torch.eye(emp_cov.size(-1))
        .unsqueeze(0)
        .unsqueeze(0)
        .repeat(emp_cov.size(0), emp_cov.size(1), 1, 1)
        .to(device)
    )

    glasso_cov = GLASSO(emp_cov, S_0, 1.0, max_iter=20000, eps=5e-5)
    print(glasso_cov[0, 0].diagonal())

    return glasso_cov


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


def get_metric(metric_type):
    METRICS = {
        "cosine": lambda gallery, query: 1.0
        - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        "euclidean": lambda gallery, query: (
            (query[:, None, :] - gallery[None, :, :]) ** 2
        ).sum(2),
        "l1": lambda gallery, query: torch.norm(
            (query[:, None, :] - gallery[None, :, :]), p=1, dim=2
        ),
        "l2": lambda gallery, query: torch.norm(
            (query[:, None, :] - gallery[None, :, :]), p=2, dim=2
        ),
    }
    return METRICS[metric_type]
