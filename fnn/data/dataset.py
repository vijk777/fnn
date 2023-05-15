import numpy as np
import pandas as pd


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

    def __init__(self, filepath, indexmap=None, transform=None):
        """
        Parameters
        ----------
        filepath : file-like object | str | pathlib.Path
            file to read
        indexmap : 1D array
            index map for the npy file
        """
        self.filepath = filepath
        self.indexmap = indexmap
        self.transform = transform

    def __getitem__(self, index):
        x = np.load(self.filepath, mmap_mode="r")
        i = index if self.indexmap is None else self.indexmap[index]
        t = np.array if self.transform is None else self.transform
        return t(x[i])


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
