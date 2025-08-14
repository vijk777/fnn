import os
import av
from tqdm import tqdm
import warnings
from typing import Union
import numpy as np
from numpy.typing import NDArray

def avi_to_numpy(
    path: Union[str, os.PathLike],
    format: str = "gray",
    *,
    show_pbar: bool = True,
) -> NDArray[np.uint8]:
    """
    Decode an AVI into a NumPy array.
    
    Args:
        path: Path to the AVI file.
        format: Format to decode frames into. Options include "gray", "rgb24", etc
            (see `av` documentation for details).
        show_pbar: Whether to show a progress bar during decoding.  

    Warnings:
        If the AVI file has a header with a different number of frames than what is decoded,
        a warning will be issued.
    Returns:
        (N, H, W) for "gray" or (N, H, W, C) for e.g. "rgb24", dtype=uint8.
    """
    frames = []
    header_total = None  # capture before container closes

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # store header frame count while stream is alive
        hf = getattr(stream, "frames", None)
        if isinstance(hf, int) and hf > 0:
            header_total = hf

        iterable = container.decode(video=0)
        if show_pbar:
            iterable = tqdm(iterable, total=header_total, desc="Decoding frames")

        for frame in iterable:
            frames.append(frame.to_ndarray(format=format))

    if not frames:
        raise RuntimeError("No frames decoded.")

    arr = np.stack(frames, axis=0)

    if header_total and header_total != arr.shape[0]:
        warnings.warn(
            f"Header reports {header_total} frames, but decoded {arr.shape[0]} frames."
        )

    return arr
