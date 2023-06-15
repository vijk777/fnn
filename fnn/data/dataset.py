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
    """Numpy File Data"""

    def __init__(self, data):
        """
        Parameters
        ----------
        data : ND array
            data array
        """
        self.fd = tempfile.mkdtemp()
        self.fp = os.path.join(self.fd, "data.npy")
        np.save(self.fp, data)

    def __getitem__(self, index):
        return np.load(self.fp, mmap_mode="r")[index]

    def __del__(self):
        os.remove(self.fp)
        os.rmdir(self.fd)


# -------------- Data Set --------------


class Dataset:
    """Data Set"""

    def __init__(self, dataframe):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            index -- data item identifier
            rows -- data items
            columns -- datainfo (`training`, `samples`) and dataitems
        """
        assert np.unique(dataframe.index).size == len(dataframe), "Index is not unique"

        for item in self.dataitems:
            assert isinstance(dataframe[item][0], Item), f"{dataframe[item][0]} is not an instance of Item"

        self.df = dataframe.astype(self.datainfo)

    @property
    def datainfo(self):
        return {"training": bool, "samples": int}

    @property
    def dataitems(self):
        return set(self.df.columns) - set(self.datainfo)

    @property
    def keys(training=True):
        """
        Parameters
        ----------
        training : bool
            training or validation

        Returns
        -------
        pandas.Index
            training or validation keys
        """
        if training:
            return self.df.index[self.df.training]
        else:
            return self.df.index[~self.df.training]

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
        data = self.df.loc[key]

        if index is None:
            return {item: data[item][:] for item in self.dataitems}
        else:
            return {item: data[item][index] for item in self.dataitems}
