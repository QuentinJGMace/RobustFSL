import torch


def L2_normalise(support, query, **kwargs):
    """
    Normalises the support and query sets using L2 normalisation.

    Args:
        support : torch.tensor of shape [batch_size, n_support, feature_dim]
        query : torch.tensor of shape [batch_size, n_query, feature_dim]
    """

    support = support / torch.norm(support, dim=-1, keepdim=True)
    query = query / torch.norm(query, dim=-1, keepdim=True)
    return support, query


def normalize_by_train_mean(support, query, train_mean, **kwargs):
    """
    Normalises the support and query sets using the mean of the training set.

    Args:
        support : torch.tensor of shape [batch_size, n_support, feature_dim]
        query : torch.tensor of shape [batch_size, n_query, feature_dim]
        train_mean : torch.tensor of shape [batch_size, feature_dim]
    """

    support = support - train_mean.unsqueeze(1)
    query = query - train_mean.unsqueeze(1)
    support /= torch.norm(support, dim=-1, keepdim=True)
    query /= torch.norm(query, dim=-1, keepdim=True)
    return support, query


def transductive_normalise(support, query, **kwargs):
    """
    Normalises the support and query sets using the mean of the support and query sets.

    Args:
        support : torch.tensor of shape [batch_size, n_support, feature_dim]
        query : torch.tensor of shape [batch_size, n_query, feature_dim]
    """

    mean_support_query = torch.cat((support, query), dim=1).mean(dim=1)
    support = support - mean_support_query.unsqueeze(1)
    query = query - mean_support_query.unsqueeze(1)
    support /= torch.norm(support, dim=-1, keepdim=True)
    query /= torch.norm(query, dim=-1, keepdim=True)
    return support, query


def paddle_normalize(support, query, **kwargs):
    dist = query.max(dim=1, keepdim=True)[0] - query.min(dim=1, keepdim=True)[0]
    dist[dist == 0.0] = 1.0
    scale = 1.0 / dist
    ratio = query.min(dim=1, keepdim=True)[0]
    query.mul_(scale).sub_(ratio)
    support.mul_(scale).sub_(ratio)
    return query, support
