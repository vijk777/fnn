import numpy as np
import pandas as pd
import os
import tempfile


# -------------- Data Item --------------


class Item:
    """Data Item"""

    def __getitem__(self, index):
        """
        Parameters
        ----------
        index : int | slice | 1D array
            used to index the data item

        Returns
        -------
        ND array
            data array
        """
        raise NotImplementedError()


class NpyFile(Item):
    """Numpy Data Item"""

    def __init__(self, filepath, indexmap=None, transform=None, dtype=None):
        """
        Parameters
        ----------
        filepath : file-like object | str | pathlib.Path
            file to read
        indexmap : 1D array
            index map for the npy file
        transform : Callable[[ND array], ND array]
            data transformation
        dtype : None | np.dtype
            data dtype
        """
        data = np.load(filepath)

        if indexmap is not None:
            data = data[indexmap]

        if transform is not None:
            data = transform(data)

        if dtype is not None:
            data = data.astype(dtype)

        self.fd = tempfile.mkdtemp()
        self.fp = os.path.join(self.fd, "data.npy")

        np.save(self.fp, data)

    def __getitem__(self, index):
        return np.load(self.fp, mmap_mode="r")[index]

    def __del__(self):
        os.remove(self.fp)
        os.rmdir(self.fd)


# -------------- Data Set --------------


class Dataset(pd.DataFrame):
    """Data Set"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert not self.datainfo - set(self.columns), "Missing data info"

        for item in self.dataitems:
            assert isinstance(self[item][0], Item), f"{self[item][0]} is not an instance of Item"

    @property
    def datainfo(self):
        return {"samples"}

    @property
    def dataitems(self):
        return set(self.columns) - self.datainfo

    def load(self, key, index=None):
        """
        Parameters
        ----------
        key : hashable
            label for DataFrame.loc
        index : int | slice | 1D array
            used to index the data item

        Returns
        -------
        dict[str, ND array]
            data items
        """
        data = self.loc[key]

        if index is None:
            return {item: data[item][:] for item in self.dataitems}
        else:
            return {item: data[item][index] for item in self.dataitems}
