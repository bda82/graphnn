import tensorflow as tf

from gns.loaders.generic_loader import GenericLoader
from gns.utils.collate_labels_disjoint import collate_labels_disjoint
from gns.utils.sp_matrices_to_sp_tensors import sp_matrices_to_sp_tensors
from gns.utils.to_disjoint import to_disjoint
from gns.utils.to_tf_signature import to_tf_signature
from gns.utils.prepend_none import prepend_none


class DisjointLoader(GenericLoader):
    """
    A Loader for disjoint mode.

    This loader represents a batch of graphs via their disjoint union.

    The loader automatically computes a batch index tensor, containing integer
    indices that map each node to its corresponding graph in the batch.

    The adjacency matrix os returned as a SparseTensor, regardless of the input.

    If `node_level=False`, the labels are interpreted as graph-level labels and
    are stacked along an additional dimension.
    If `node_level=True`, then the labels are stacked vertically.

    Notes:
        TensorFlow 2.4 or above is required to use this Loader's `load()` method in a Keras training loop.

    Output parameters:

    For each batch, returns a tuple `(inputs, labels)`.

    Input parameters::
        `x`: node attributes of shape `[n_nodes, n_node_features]`;
        `a`: adjacency matrices of shape `[n_nodes, n_nodes]`;
        `e`: edge attributes of shape `[n_edges, n_edge_features]`;
        `i`: batch index of shape `[n_nodes]`.

    Notes:
        `labels` have shape `[batch, n_labels]` if `node_level=False` or `[n_nodes, n_labels]` otherwise.
    """

    def __init__(
        self, dataset, node_level=False, batch_size=1, epochs=None, shuffle=True
    ):
        """
        Args:
            `dataset`: a graph Dataset;
            `node_level`: bool, if `True` stack the labels vertically for node-level prediction;
            `batch_size`: size of the mini-batches;
            `epochs`: number of epochs to iterate over the dataset. By default (`None`) iterates indefinitely;
            `shuffle`: whether to shuffle the data at the start of each epoch.
        """
        self.node_level = node_level
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_disjoint(y, node_level=self.node_level)

        output = to_disjoint(**packed)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return output, y

    def load(self):
        return tf.data.Dataset.from_generator(
            lambda: self, output_signature=self.tf_signature()
        )

    def tf_signature(self):
        """
        Adjacency matrix has shape `[n_nodes, n_nodes]`
        Node features have shape `[n_nodes, n_node_features]`
        Edge features have shape `[n_edges, n_edge_features]`
        Targets have shape `[*, n_labels]`
        """
        signature = self.dataset.signature
        if "y" in signature:
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])
        if "a" in signature:
            signature["a"]["spec"] = tf.SparseTensorSpec

        signature["i"] = dict()
        signature["i"]["spec"] = tf.TensorSpec
        signature["i"]["shape"] = (None,)
        signature["i"]["dtype"] = tf.as_dtype(tf.int64)

        return to_tf_signature(signature)


def disjoint_loader_fabric(
    dataset,
    node_level=False,
    batch_size=1,
    epochs=None,
    shuffle=True
):
    return DisjointLoader(
        dataset,
        node_level,
        batch_size,
        epochs,
        shuffle
    )
