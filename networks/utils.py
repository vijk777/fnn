import torch
import torch.nn.functional as F

from typing import Optional


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


def rmat_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
):
    """
    Args:
        x (torch.Tensor): shape = [N]
        y (torch.Tensor): shape = [N]
        z (torch.Tensor): shape = [N]
    Returns:
        (torch.Tensor): shape = [N, 3, 3]
    """
    N = len(x)
    assert N == len(y) == len(z)

    A = torch.eye(3, device=x.device).repeat(N, 1, 1)
    B = torch.eye(3, device=x.device).repeat(N, 1, 1)
    C = torch.eye(3, device=x.device).repeat(N, 1, 1)

    cos_z = torch.cos(z)
    sin_z = torch.sin(z)

    A[:, 0, 0] = cos_z
    A[:, 0, 1] = -sin_z
    A[:, 1, 0] = sin_z
    A[:, 1, 1] = cos_z

    cos_y = torch.cos(y)
    sin_y = torch.sin(y)

    B[:, 0, 0] = cos_y
    B[:, 0, 2] = sin_y
    B[:, 2, 0] = -sin_y
    B[:, 2, 2] = cos_y

    cos_x = torch.cos(x)
    sin_x = torch.sin(x)

    C[:, 1, 1] = cos_x
    C[:, 1, 2] = -sin_x
    C[:, 2, 1] = sin_x
    C[:, 2, 2] = cos_x

    return A.matmul(B).matmul(C)


def isotropic_grid_2d(
    height: int,
    width: int,
    device: Optional[str] = None,
):
    """
    Args:
        height  (int)
        width   (int)
    Returns:
        (torch.Tensor): shape = [height, width, 2]
    """
    h = torch.linspace(-1, 1, height)
    w = torch.linspace(-1, 1, width)

    if height < width:
        h = h * height / width
        scale = (width - 1) / width

    else:
        w = w * width / height
        scale = (height - 1) / height

    return torch.stack(torch.meshgrid(w * scale, h * scale, indexing="xy"), dim=2).to(device=device)
