import numpy as np
import pandas as pd
from multiprocessing import Process, Queue


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


# -------------- Data Index --------------


class Index:
    """Data Index"""

    def __call__(self, samples, rng=None):
        """
        Parameters
        ----------
        samples : int
            number of samples
        rng : np.random.Generator | None
            random number generator

        Returns
        -------
        1D array | None
            data index
        """
        return


class SubsampleIndex(Index):
    """Subsample Data Index"""

    def __init__(self, samplesize):
        """
        Parameters
        ----------
        samplesize : int
            sample size
        """
        self.samplesize = samplesize

    def __call__(self, samples, rng=None):
        assert samples >= self.samplesize, "Not enough samples"

        choice = np.random.choice if rng is None else rng.choice
        return choice(samples - self.samplesize) + np.arange(self.samplesize)


# -------------- Data Set --------------


class Data(pd.DataFrame):
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

    def batches(self, batchsize, index, keys, batches=None, seed=42):
        """
        Parameters
        ----------
        batchsize : int
            batch size
        index : fnn.data.Index
            data index generator
        keys : list[hashable]
            labels for DataFrame.loc
        batches : int | None
            number of batches -- int : random batches | None : ordered batches (until all keys are yielded)
        seed : int
            random number generation

        Yields
        ------
        dict[str, ND array]
            data batches stacked on dim=0
        """
        rng = np.random.default_rng(seed)

        if batches is None:
            size = len(keys)
        else:
            size = batchsize * batches
            replace = size > len(keys)
            keys = rng.choice(keys, size=size, replace=replace)

        indexes = [index(samples, rng) for samples in self.loc[keys].samples]

        d = {item: [] for item in self.dataitems}

        q = Queue(batchsize)

        p = Process(target=self._load, args=(q, keys, indexes))
        p.start()

        for b in range(size):

            for k, v in q.get().items():
                d[k].append(v)

            if (b + 1) % batchsize:
                continue

            batch = {}
            for k, v in d.items():
                batch[k] = np.stack(v)
                v.clear()
            yield batch

        p.join()

        if (b + 1) % batchsize:
            yield {k: np.stack(v) for k, v in d.items()}

    def _load(self, queue, keys, indexes):
        for key, index in zip(keys, indexes):
            data = self.load(key, index)
            queue.put(data)
