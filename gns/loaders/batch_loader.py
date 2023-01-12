import tensorflow as tf

from gns.loaders.generic_loader import GenericLoader
from gns.utils.collate_labels_batch import collate_labels_batch
from gns.utils.sp_matrices_to_sp_tensors import sp_matrices_to_sp_tensors
from gns.utils.to_batch import to_batch
from gns.utils.to_tf_signature import to_tf_signature


class BatchLoader(GenericLoader):
    """
    A Loader for batch mode.

    BatchLoader returns batches of graphs stacked along an extra dimension,
    with all "node" dimensions padded to be equal among all graphs.

    If `n_max` is the number of nodes of the biggest graph in the batch, then
    the padding consist of adding zeros to the node features, adjacency matrix,
    and edge attributes of each graph so that they have shapes
        `[n_max, n_node_features]`,
        `[n_max, n_max]`,
        `[n_max, n_max, n_edge_features]`.

    The zero-padding is done batch-wise, which saves up memory at the cost of
    more computation.
    Notes:
        The adjacency matrix and edge attributes are returned as dense arrays.

    if `mask=True`, node attributes will be extended with a binary mask that indicates
    valid nodes (the last feature of each node will be 1 if the node was originally in
    the graph and 0 if it is a fake node added by zero-padding).

    If `node_level=False`, the labels are interpreted as graph-level labels and
    are returned as an array of shape `[batch, n_labels]`.

    If `node_level=True`, then the labels are padded along the node dimension and are
    returned as an array of shape `[batch, n_max, n_labels]`.

    Input parameters:
        `dataset`: a graph Dataset;
        `mask`: bool, whether to add a mask to the node features;
        `batch_size`: int, size of the mini-batches;
        `epochs`: int, number of epochs to iterate over the dataset. None by default - iterates indefinitely;
        `shuffle`: bool, whether to shuffle the data at the start of each epoch;
        `node_level`: bool, if `True` pad the labels along the node dimension;

    Output parameters:
        For each batch, returns a tuple '(inputs, labels)'. 'inputs is a tuple containing:
        'x': node attributes of shape '[batch, n_max, n_node_features]';
        'a': adjacency matrices of shape '[batch, n_max, n_max]';
        'e': edge attributes of shape '[batch, n_max, n_max, n_edge_features]'.
             'labels' have shape '[batch, n_labels]' if 'node_level=False' or
             '[batch, n_max, n_labels]' otherwise.
    """

    def __init__(
        self,
        dataset,
        mask=False,
        batch_size=1,
        epochs=None,
        shuffle=True,
        node_level=False,
    ):
        self.mask = mask
        self.node_level = node_level
        self.signature = dataset.signature
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_batch(y, node_level=self.node_level)

        output = to_batch(**packed, mask=self.mask)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            return output, y

    def prepend_none(self, t):
        return (None,) + t

    def tf_signature(self):
        """
        Adjacency matrix has shape `[batch, n_nodes, n_nodes]`
        Node features have shape `[batch, n_nodes, n_node_features]`
        Edge features have shape `[batch, n_nodes, n_nodes, n_edge_features]`
        Labels have shape `[batch, n_labels]`.
        """
        signature = self.signature

        for k in signature:
            signature[k]["shape"] = self.prepend_none(signature[k]["shape"])

        if "x" in signature and self.mask:
            # In case we have a mask, the mask is concatenated to the features

            signature["x"]["shape"] = signature["x"]["shape"][:-1] + (
                signature["x"]["shape"][-1] + 1,
            )

        if "a" in signature:
            # Adjacency matrix in batch mode is dense

            signature["a"]["spec"] = tf.TensorSpec

        if "e" in signature:
            # Edge attributes have an extra None dimension in batch mode

            signature["e"]["shape"] = self.prepend_none(signature["e"]["shape"])

        if "y" in signature and self.node_level:
            # Node labels have an extra None dimension

            signature["y"]["shape"] = self.prepend_none(signature["y"]["shape"])

        return to_tf_signature(signature)


def batch_loader_fabric(
    dataset, mask=False, batch_size=1, epochs=None, shuffle=True, node_level=False
):
    return BatchLoader(dataset, mask, batch_size, epochs, shuffle, node_level)
