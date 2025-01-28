import os
import hashlib
import zipfile
import requests
import torch
import pandas as pd
from tqdm import tqdm


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


def download(url, file_path, chunk_size=8192, verbose=True):
    """
    Parameters
    ----------
    url : str
        source url
    file_path : os.PathLike
        destination file path
    chunk_size : int
        size of the chunks (in bytes) to read

    Returns
    -------
    str
        MD5 checksum of downloaded file as a hexadecimal string
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    size = int(response.headers.get("content-length", 0))

    with open(file_path, "wb") as file:
        with tqdm(total=size, unit="B", unit_scale=True, disable=not verbose) as bar:
            for chunk in response.iter_content(chunk_size):
                file.write(chunk)
                bar.update(len(chunk))

    return md5sum(file_path, chunk_size)


def unzip(file_path, directory=None):
    """
    Parameters
    ----------
    file_path : os.PathLike
        path to the zip file
    directory : os.PathLike
        destination directory
    """
    with zipfile.ZipFile(file_path, "r") as zf:
        zf.extractall(directory)


def params(session, scan_idx, directory):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    directory : os.PathLike
        data directory

    Returns
    -------
    Dict[str, torch.Tensor]
        model parameters
    """

    def load(path):
        return torch.load(os.path.join(directory, path), map_location="cpu")

    return dict(**load("params_core.pt"), **load(f"params_{session}_{scan_idx}.pt"))


def units(session, scan_idx, directory):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    directory : os.PathLike
        data directory

    Returns
    -------
    int
        number of units
    """
    path = os.path.join(directory, "scans.csv")
    df = pd.read_csv(path).set_index(["session", "scan_idx"])
    try:
        return int(df.loc[session, scan_idx].units)
    except KeyError:
        raise ValueError(f"Scan {session}-{scan_idx} not found.")


def unit_ids(session, scan_idx, directory):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    directory : os.PathLike
        data directory

    Returns
    -------
    pd.DataFrame
        dataframe mapping readout ids to unit ids
    """
    path = os.path.join(directory, "units.csv")
    df = pd.read_csv(path).groupby(["session", "scan_idx"])
    try:
        return df.get_group((session, scan_idx)).set_index("readout_id")
    except KeyError:
        raise ValueError(f"Scan {session}-{scan_idx} not found.")
