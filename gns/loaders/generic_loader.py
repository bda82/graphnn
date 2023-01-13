from typing import Any, Generator

import numpy as np

from gns.utils.batch_generator import batch_generator
from gns.utils.to_tensorflow_signature import to_tensorflow_signature


class GenericLoader:
    """
    The base class for the dataset loader.

    It is intended for iteration on dataset and return of packets (batches) of graphs for Keras models.
    It works on the principle of a python generator (returns 1 value per 1 request).

    This is achieved by having a generator object that creates lists of graphs.,
    which are then mapped together and returned as tensors inside Keras.

    The core of the loader is the `batch` method.
    It takes as input a list of Graph objects and returns a list of tensors,
    arrays of Numpy `np.arrays`, or SparseTensors.

    Notes:
    For example, if all graphs have the same number of nodes and size attributes, a simple sorting function could be:

    ```py
    def collate(self, batch):
        x = np.array([g.x for g in batch])
        a = np.array([g.a for g in batch)]
        return x, a
    ```

    The load() method returns an object that can be passed to the Keras model when using the functions:
        fit
        predict
        evaluate

    We can use it as follows:
    ``py
    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch)
    ```

    The steps_per_epoch property represents the number of batches in each epoch and is a mandatory parameter for working in modes:
        fit
        predict
        evaluate

    Notes:
        If a proprietary learning function is used, we can describe the input signature of the batches using a type system
        TensorFlow: tf.TypeSpec, to eliminate unnecessary retrace.
        The signature is automatically calculated by calling the loader.tf_signature() function.

    For example, a simple learning step can be written as:

    ```py
    @tf.function(input_signature=loader.tf_signature()) # The signature is defined here
    de train_step(inputs, target):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            loss = loss_fn(target, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ```

    Then we can train our model in a loop as follows:

    ```py
    for batch in loader:
        train_step(*batch)
    ```
    """

    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        """
        Args:
            dataset: a dataset object of type gns.dataset.Dataset
            batch_size: int, the size of the minimum batch;
            epochs`: int, the number of epochs to iterate over the dataset (by default None - repeats infinitely)
            shuffle`: bool, a sign of the need to shuffle the data set at the beginning of each epoch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self._generator = self.generator()

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def generator(self) -> Generator[list, Any, Any]:
        """
        Returns a list of butches (packages) of the Graph object.

        Returns:

        """
        return batch_generator(
            self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
        )

    def collate(self, batch):
        """
        Converts a list of graph objects into tensors or Numpy np.array arrays representing a batch (butch).

        Args:
            batch: list pf Graph objects

        Returns:

        """
        raise NotImplementedError

    def load(self):
        """
        Returns an object that can be passed to the Keras model when the function is used:
            fit
            predict
            evaluate

        By default, it returns a reference to itself as a generator.

        Returns:

        """
        return self

    def tf_signature(self):
        """
        Returns the signature of the mapped packets using the tf.Type Spec type system.

        By default, the signature is the signature of the dataset.signature database:
            the adjacency matrix has the form `[n_nodes, n_nodes]`
            node features have the form `[n_nodes, n_node_features]`
            edge features have the form `[n_edges, n_node_features]`
            labels have the form `[..., n_labels]`

        Returns:

        """
        signature = self.dataset.signature
        return to_tensorflow_signature(signature)

    def pack(self, batch) -> dict:
        """
        Groups the attributes of a graph package into separate lists, and they themselves are in the dictionary.

        For example:

        If we have a package of three graphs g1, g2 and g3 with node singularities (x1, x2, x3) and
        adjacency matrix (a1, a2, a3), this method will return a dictionary:
        ```py
        >>> {'a_list': [a1, a2, a3], 'x_list': [x1, x2, x3]}
        ```
        Args:
            batch: Graph list

        Returns:

        """
        output = [list(elem) for elem in zip(*[g.numpy() for g in batch])]
        keys = [k + "_list" for k in self.dataset.signature.keys()]

        return dict(zip(keys, output))

    @property
    def steps_per_epoch(self):
        """
        Returns the number of all self-sized packets.batch_size in the dataset, that is, how many packets are in the epoch.

        Returns:

        """
        return int(np.ceil(len(self.dataset) / self.batch_size))


def generic_loader_fabric(dataset, batch_size=1, epochs=None, shuffle=True):
    return GenericLoader(dataset, batch_size, epochs, shuffle)
