import os
import tempfile
import logging
from fnn.microns.load import download, unzip, params, units, unit_ids
from fnn.microns.build import network

URL = "https://www.dropbox.com/scl/fi/fwhuovi16vqgkhymofssv/microns.zip?rlkey=j1enuhov22rvk0b0lkzl8ynw5&st=qne085vd&dl=1"
MD5 = "58fcac4b31ad2902c81e339432cec787"

logger = logging.getLogger("fnn.microns")
logger.addHandler(logging.StreamHandler())


def scan(session, scan_idx, directory=None, verbose=True):
    """
    Parameters
    ----------
    session : int
        scan session
    scan_idx : int
        scan index
    directory : os.PathLike | None
        directory for model parameters and metadata. defaults to current working directory
    verbose : bool
        display data download progress

    Returns
    -------
    fnn.model.networks.Visual
        predictive model of the experimental scan
    pd.DataFrame
        dataframe mapping readout ids to unit ids
    """
    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    directory = os.path.join(directory or os.getcwd(), "fnn", "microns")
    load = lambda f: f(session, scan_idx, directory)

    if not os.path.exists(directory):
        logger.info(f"Downloading model parameters and metadata to `{directory}`")

        with tempfile.TemporaryDirectory() as tmpdir:
            zipfile = os.path.join(tmpdir, "microns.zip")

            md5 = download(URL, zipfile, verbose=verbose)
            assert md5 == MD5, f"md5 for downloaded file is {md5}, expected {MD5}"

            unzip(zipfile, directory)

    model = network(load(units))
    model.load_state_dict(load(params))

    metadata = load(unit_ids)

    return model, metadata
