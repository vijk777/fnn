import os
import hashlib
import zipfile
import torch
import pandas as pd


def directory(root=None):
    """
    Parameters
    ----------
    root : os.PathLike
        root directory

    Returns
    -------
    str
        microns data directory
    """
    return os.path.join(root or os.getcwd(), "microns")


def md5sum(file_path, chunk_size=8192):
    """
    Parameters
    ----------
    file_path : os.PathLike
        path to the file
    chunk_size : int
        size of the chunks (in bytes) to read

    Returns
    -------
    str
        MD5 checksum as a hexadecimal string
    """
    md5 = hashlib.md5()

    with open(file_path, "rb") as file:
        while chunk := file.read(chunk_size):
            md5.update(chunk)

    return md5.hexdigest()


def unzip(file_path, target=None):
    """
    Parameters
    ----------
    file_path : os.PathLike
        path to the zip file
    target : os.PathLike
        target root directory
    """
    with zipfile.ZipFile(file_path, "r") as zf:
        zf.extractall(directory(target))


def params(session, scan_idx, root=None):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    root : os.PathLike
        root directory

    Returns
    -------
    Dict[str, torch.Tensor]
        model parameters
    """
    microns_dir = directory(root)

    def load(path):

        return torch.load(os.path.join(microns_dir, path), map_location="cpu")

    return dict(**load("params_core.pt"), **load(f"params_{session}_{scan_idx}.pt"))


def units(session, scan_idx, root=None):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    root : os.PathLike
        root directory

    Returns
    -------
    int
        number of units
    """
    path = os.path.join(directory(root), "scans.csv")
    df = pd.read_csv(path).set_index(["session", "scan_idx"])
    try:
        return int(df.loc[session, scan_idx].units)
    except KeyError:
        raise ValueError(f"Scan {session}-{scan_idx} not found.")


def unit_ids(session, scan_idx, root=None):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    root : os.PathLike
        root directory

    Returns
    -------
    pd.DataFrame
        dataframe mapping readout ids to unit ids
    """
    path = os.path.join(directory(root), "units.csv")
    df = pd.read_csv(path).groupby(["session", "scan_idx"])
    try:
        return df.get_group((session, scan_idx)).set_index("readout_id")
    except KeyError:
        raise ValueError(f"Scan {session}-{scan_idx} not found.")
