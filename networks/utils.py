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
    dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
):
    """
    Args:
        height  (int)
        width   (int)
    Returns:
        (torch.Tensor): shape = [height, width, 2]
    """
    grid_x = torch.linspace(-1, 1, width, dtype=dtype, device=device)
    grid_y = torch.linspace(-1, 1, height, dtype=dtype, device=device)

    if height < width:
        grid_y = grid_y * height / width
    else:
        grid_x = grid_x * width / height

    grid = torch.meshgrid(
        grid_x,
        grid_y,
        indexing="xy",
    )
    return torch.stack(grid, dim=2)


def isotropic_grid_sample_2d(
    x: torch.Tensor,
    grid: torch.Tensor,
    pad_mode: str = "constant",
    pad_value: float = 0,
):
    """
    Args:
        x       (torch.Tensor)  : shape = [n, c, h, w]
        grid    (torch.Tensor)  : shape = [n, h', w', 2]
        pad_mode         (str)  : 'constant' or 'replicate'
        pad_value      (float)  : value used when pad_mode=='constant'
    Returns:
        (torch.Tensor)          : shape = [n, c, h', w']
    """
    if pad_mode == "constant":
        x = x - pad_value
        finalize = lambda x: x + pad_value
        padding_mode = "zeros"

    elif pad_mode == "replicate":
        if pad_value:
            raise ValueError("cannot specify pad_value with pad_mode='pad_value'")
        finalize = lambda x: x
        padding_mode = "border"

    else:
        raise ValueError("pad_mode must either be 'constant' or 'replicate'")

    grid_x, grid_y = grid.unbind(dim=3)

    _, _, height, width = x.shape
    if height < width:
        grid_y = grid_y * width / height
    else:
        grid_x = grid_x * height / width

    _, height, width, _ = grid.shape
    grid = [
        grid_x * (width - 1) / width,
        grid_y * (height - 1) / height,
    ]

    x = F.grid_sample(
        x,
        grid=torch.stack(grid, dim=3),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    return finalize(x)
