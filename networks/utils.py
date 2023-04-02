import torch
import torch.nn.functional as F


def to_groups_2d(
    tensor: torch.Tensor,
    groups: int,
):
    """
    Args:
        tensor (torch.Tensor): shape = [n, c, h, w]
        groups (float)
    Returns:
        (torch.Tensor): shape = [n, groups, c // groups, h, w]
    """
    N, _, H, W = tensor.shape
    return tensor.view(N, groups, -1, H, W)
