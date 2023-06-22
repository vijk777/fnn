import torch
from torch.nn import functional


def to_groups_2d(tensor, groups):
    """Reshapes a 2D (H, W) tensor to channel groups

    Parameters
    ----------
    tensor : Tensor
        [N, C, H, W]
    groups : int
        channel groups (G)

    Returns
    -------
    Tensor
        [N, G, C//G, H, W]
    """
    N, _, H, W = tensor.shape
    return tensor.view(N, groups, -1, H, W)


def cat_groups_2d(tensors, groups):
    """Groupwise concatenation of 2D (H, W) tensors along the channel dimension (C)

    Parameters
    ----------
    tensors : Sequence[Tensor]
        [[N, C, H, W], ...]
    groups : int
        channel groups (G)

    Returns
    -------
    Tensor
        [N, C', H, W]
    """
    tensors = [to_groups_2d(tensor, groups) for tensor in groups]
    return torch.cat(tensors, 2).flatten(1, 2)


def rmat_3d(x, y, z):
    """Creates a 3D rotation matrix

    Parameters
    ----------
    x : Tensor
        [N], angle (radians) of rotation around x axis
    y : Tensor
        [N], angle (radians) of rotation around y axis
    z : Tensor
        [N], angle (radians) of rotation around z axis

    Returns
    -------
    Tensor
        [N, 3, 3]
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


def isotropic_grid_2d(height, width, major="x", dtype=None, device=None):
    """Creates an isotropic 2d grid

    Parameters
    ----------
    height : int
        grid height (H)
    width : int
        grid width (W)
    major : str
        "x" | "y" , axis by which grid is scaled
    dtype : torch.dtype | None
        grid dtype
    device : torch.device | None
        grid device

    Returns
    -------
    Tensor
        [H, W, 2]
    """
    grid_x = torch.linspace(-1, 1, width, dtype=dtype, device=device)
    grid_y = torch.linspace(-1, 1, height, dtype=dtype, device=device)

    if major == "x":
        grid_y = grid_y * height / width
        scale = (width - 1) / width

    elif major == "y":
        grid_x = grid_x * width / height
        scale = (height - 1) / height

    else:
        raise ValueError("major must be either 'x' or 'y'")

    grid = torch.meshgrid(
        grid_x * scale,
        grid_y * scale,
        indexing="xy",
    )
    return torch.stack(grid, dim=2)


def isotropic_grid_sample_2d(x, grid, major="x", pad_mode="constant", pad_value=0):
    """Isotropic 2D sampling

    Parameters
    ----------
    x : Tensor
        [N, C, H, W]
    grid : Tensor
        [N, H', W', 2]
    major : str
        "x" | "y" , axis by which sampling is scaled
    pad_mode : str
        "constant" | "replicate" , padding mode for out-of-bounds grid values
    pad_value : float
        value of padding when pad_mode=="constant"

    Returns
    -------
    Tensor
        [N, C, H', W']
    """
    if pad_mode == "constant":
        x = x - pad_value
        finalize = lambda x: x + pad_value
        padding_mode = "zeros"

    elif pad_mode == "replicate":
        if pad_value:
            raise ValueError("cannot specify pad_value with pad_mode='constant'")
        finalize = lambda x: x
        padding_mode = "border"

    else:
        raise ValueError("pad_mode must either be 'constant' or 'replicate'")

    grid_x, grid_y = grid.unbind(dim=3)

    _, _, height, width = x.shape
    if major == "x":
        grid_y = grid_y * width / height
        scale = width / (width - 1)

    elif major == "y":
        grid_x = grid_x * height / width
        scale = height / (height - 1)

    else:
        raise ValueError("major must be either 'x' or 'y'")

    _, height, width, _ = grid.shape
    grid = [
        grid_x * scale * (width - 1) / width,
        grid_y * scale * (height - 1) / height,
    ]

    x = functional.grid_sample(
        x,
        grid=torch.stack(grid, dim=3),
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )
    return finalize(x)
