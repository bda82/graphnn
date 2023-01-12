import tensorflow as tf

from gns.loaders.generic_loader import GenericLoader
from gns.utils.collate_labels_disjoint import collate_labels_disjoint
from gns.utils.sp_matrices_to_sp_tensors import sp_matrices_to_sp_tensors
from gns.utils.to_disjoint import to_disjoint


class SingleLoader(GenericLoader):
    """
    Instance of a single loader.

    This loader creates tensors representing a single graph. As such, it can used only with datasets of length 1.

    Notes:
        The batch_size parameter cannot be set.

    The loader supports sample weights using the `sample_weights` argument.

    If set, each packet will be a tuple (inputs, labels, sample_weights).

    Input parameters:
        `dataset`: instance of gns.dataset.Dataset with only one graph
        `epochs`: int, the number of epochs to iterate over the dataset (by default None - infinite iteration)
        `shuffle`: bool, a sign of the need to shuffle data at the start of each epoch
        `sample_weights`: Numpy array, added to the output automatically

    Output parameters:
        Returns a python tuple of the format (inputs, labels) or (inputs, labels, sample_weights)

    `inputs` is a tuple containing graph data matrices only if they are not None:
        `x`: same as `dataset[0].x`;
        `a`: same as `dataset[0].a` (sparse scipy matrices will be converted to SparseTensors)
        `e`: the same as `dataset[0].e`
        `labels`: the same as `dataset[0].y`
        `sample_weights`: the same as just an array passed when creating the loader
    """

    def __init__(self, dataset, epochs=None, sample_weights=None):
        if len(dataset) != 1:
            raise ValueError(
                "`SingleLoader` it can only be used with a dataset that has a unit graph."
            )

        self.sample_weights = sample_weights

        super().__init__(dataset, batch_size=1, epochs=epochs, shuffle=False)

    def collate(self, batch):
        """
        Loader core.

        Args:
            batch: batch of data

        Returns: list of tensors

        """
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=True)

        output = to_disjoint(**packed)
        output = output[:-1]
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        output = (output,)

        if y is not None:
            output += (y,)

        if self.sample_weights is not None:
            output += (self.sample_weights,)

        if len(output) == 1:
            output = output[0]

        return output

    def load(self):
        output = self.collate(self.dataset)

        return tf.data.Dataset.from_tensors(output).repeat(self.epochs)


def single_loader_fabric(dataset, epochs=None, sample_weights=None):
    return SingleLoader(dataset, epochs, sample_weights)
