import tensorflow as tf

from gns.scatter.mixed_mode_support import mixed_mode_support


@mixed_mode_support
def scatter_min(messages, indices, n_nodes):
    """
    Reduces the number of messages according to the segments defined by the "indexes", with
    support for messages in single/disjoint mode (single/disjoint) (rank 2) and mixed mode (mixed) (rank 3).
    The output has the same rank as the input, with the dimension "nodes" changed to the value of the n_nodes parameter.

    For single/disjoint mode, messages are expected to have the form:
        `[n_messages, n_features]` and outputs should have the same form
        `[n_nodes, n_features]`

    For mixed mode, messages are expected to have the form
        `[batch, n_messages, n_features]` and outputs should have the same form
        `[batch, n_nodes, n_features]`

    It is expected that the indexes will always be a 1-dimensional tensor of integers <n_nodes>, with
    the form `[n_messages]`

    For any missing index (i.e. any integer within 0 <= i < n_nodes that is not
    displayed in indexes) the corresponding output will be the minimum possible value for the message type.
    If this index i is negative, it is ignored during aggregation.

    Args:
        messages: two-dimensional (2D) or three-dimensional (3D) tensor
        indices: one-dimensional tensor with intexes in the dimension of message nodes
        n_nodes: measurement of output data by the dimension of nodes
    
    Returns:
        tensor with the same rank as messages
    """
    return tf.math.unsorted_segment_min(messages, indices, n_nodes)
