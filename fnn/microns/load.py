import os
import torch
import pandas as pd


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
