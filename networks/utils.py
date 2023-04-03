import torch
import torch.nn.functional as F


def to_groups_2d(
    tensor: torch.Tensor,
    groups: int,
):
    """
    Args:
        tensor  (torch.Tensor)  : shape = [n, c, h, w]
        groups  (float)
    Returns:
        (torch.Tensor): shape = [n, groups, c // groups, h, w]
    """
    N, _, H, W = tensor.shape
    return tensor.view(N, groups, -1, H, W)


def rmat_3d(alpha, beta, gamma):
    """
    Args:
        alpha   (torch.Tensor)  : shape = [N]
        beta    (torch.Tensor)  : shape = [N]
        gamma   (torch.Tensor)  : shape = [N]
    Returns:
        (torch.Tensor): shape = [N, 3, 3]
    """
    N = len(alpha)
    assert N == len(beta) == len(gamma)

    A = torch.eye(3, device=alpha.device).repeat(N, 1, 1)
    B = torch.eye(3, device=alpha.device).repeat(N, 1, 1)
    C = torch.eye(3, device=alpha.device).repeat(N, 1, 1)

    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    A[:, 0, 0] = cos_alpha
    A[:, 0, 1] = -sin_alpha
    A[:, 1, 0] = sin_alpha
    A[:, 1, 1] = cos_alpha

    B[:, 0, 0] = cos_beta
    B[:, 0, 2] = sin_beta
    B[:, 2, 0] = -sin_beta
    B[:, 2, 2] = cos_beta

    C[:, 1, 1] = cos_gamma
    C[:, 1, 2] = -sin_gamma
    C[:, 2, 1] = sin_gamma
    C[:, 2, 2] = cos_gamma

    return A.matmul(B).matmul(C)
