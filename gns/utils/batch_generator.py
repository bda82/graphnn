from typing import Any, Generator

import numpy as np

from gns.utils.shuffle_inplace import shuffle_inplace


def batch_generator(
    data, batch_size=32, epochs=None, shuffle=True
) -> Generator[list, Any, Any]:
    """
    Batch generator.
    Iterates over data with a given number of epochs, returns packages with a batch_size limit as a python generator (yield)
    by one value.

    Args:
        data: Numpy array np.array or a list of such arrays np.arrays with the same first dimension
        batch_size: the number of samples in the batch (batch)
        epochs: the number of attempts to iterate on the data (by default None -iterate infinitely);
        shuffle: there is no need to shuffle data before the start of the epoch

    Returns:
        butch of a given size batch_size
    """
    if not isinstance(data, (list, tuple)):
        data = [data]
    if len(data) < 1:
        raise ValueError("Данные не могут быть пустыми")
    if len({len(item) for item in data}) > 1:
        raise ValueError("Все входные данные должны иметь одинаковый размер (__len__)")

    if epochs is None or epochs == -1:
        epochs = np.inf
    len_data = len(data[0])
    batches_per_epoch = int(np.ceil(len_data / batch_size))
    epoch = 0
    while epoch < epochs:
        epoch += 1
        if shuffle:
            shuffle_inplace(*data)
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            stop = min(start + batch_size, len_data)
            to_yield = [item[start:stop] for item in data]
            if len(data) == 1:
                to_yield = to_yield[0]

            yield to_yield
