import numpy as np
from tqdm import tqdm
from torch import randint
from torch.multiprocessing import spawn, Queue


# -------------- Loader Prototype --------------


class Loader:
    """Data Loader"""

    def _init(self, dataset):
        """
        Parameters
        ----------
        dataset : fnn.data.dataset.Dataset
            dataset to load
        """
        self.dataset = dataset

    def __call__(self, training=True, display_progress=True):
        """
        Parameters
        ----------
        training : bool
            training or validation
        display_progress : bool
            display progress

        Yields
        -------
        dict
            training or validation data
        """
        raise NotImplementedError()


# -------------- Loader Types --------------


class Batches(Loader):
    """Randomly Sampled Batches"""

    def __init__(self, sample_size, batch_size, training_size, validation_size):
        """
        Parameters
        ----------
        sample_size : int
            number of samples in a datapoint
        batch_size : int
            number of datapoints in a batch
        training_size : int
            number of training batches in an epoch
        validation_size : int
            number of validation batches in an epoch
        """
        assert sample_size > 0
        assert batch_size > 0

        self.sample_size = int(sample_size)
        self.batch_size = int(batch_size)
        self.training_size = int(training_size)
        self.validation_size = int(validation_size)

    def _init(self, dataset):
        """
        Parameters
        ----------
        dataset : fnn.data.dataset.Dataset
            dataset to load
        """
        assert dataset.df.samples.min() >= self.sample_size
        super()._init(dataset)

    def _random_keys(self, training=True):
        if training:
            keys = self.dataset.keys(training=True)
            size = self.batch_size * self.training_size
        else:
            keys = self.dataset.keys(training=False)
            size = self.batch_size * self.validation_size

        if not len(keys) or not size:
            return []
        else:
            idx = randint(high=len(keys), size=(size,)).numpy()
            return keys[idx].tolist()

    def _random_indexes(self, key):
        high = self.dataset.df.loc[key].samples - self.sample_size
        if high > 0:
            return randint(high=high, size=(1,)).item() + np.arange(self.sample_size)
        else:
            return np.arange(self.sample_size)

    def _load(self, i, queue, keys, indexes):
        assert i == 0
        for key, index in zip(keys, indexes):
            data = self.dataset.load(key, index)
            queue.put(data)

    def __call__(self, training=True, display_progress=True):
        """
        Parameters
        ----------
        training : bool
            training or validation
        display_progress : bool
            display progress

        Yields
        -------
        dict
            training or validation data
        """
        keys = self._random_keys(training)

        if keys:
            indexes = [self._random_indexes(key) for key in keys]
        else:
            return

        d = {item: [] for item in self.dataset.dataitems}
        q = Queue(self.batch_size)
        c = spawn(self._load, args=(q, keys, indexes), nprocs=1, join=False)

        if display_progress:
            if training:
                iterbar = tqdm(desc="Training Batches", total=self.training_size)
            else:
                iterbar = tqdm(desc="Validation Batches", total=self.validation_size)

        for b, _ in enumerate(keys):

            for k, v in q.get().items():
                d[k].append(v)

            if (b + 1) % self.batch_size:
                continue

            batch = {}
            for k, v in d.items():
                batch[k] = np.stack(v, axis=1)
                v.clear()

            if display_progress:
                iterbar.update(n=1)

            yield batch

        assert c.join()
