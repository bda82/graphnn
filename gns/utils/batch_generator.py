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
    # assert parameters

    if not isinstance(data, (list, tuple)):
        data = [data]

    if len(data) < 1:
        raise ValueError("Data should not be empty.")

    if len({len(item) for item in data}) > 1:
        raise ValueError("All inputs should have the same length (__len__).")

    if epochs is None or epochs == -1:
        epochs = np.inf

    batches_per_epoch_count = int(np.ceil(len(data[0]) / batch_size))

    epoch_number = 0
    while epoch_number < epochs:
        epoch_number += 1

        if shuffle:
            shuffle_inplace(*data)

        for batch in range(batches_per_epoch_count):
            start = batch * batch_size
            stop = min(start + batch_size, len(data[0]))
            yield_generate = [item[start:stop] for item in data]
            if len(data) == 1:
                yield_generate = yield_generate[0]

            yield yield_generate
