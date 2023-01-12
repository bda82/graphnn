import tensorflow as tf

from gns.layer.global_pool import GlobalPoolLayer


class GlobalAveragePool(GlobalPoolLayer):
    """
    An average pooling layer.
    Pools a graph by computing the average of its node features.

    Modes:
        single
        disjoint
        mixed
        batch

    Input parameters:
        Node features of shape `([batch], n_nodes, n_node_features)`
        Graph IDs of shape `(n_nodes, )` (only in disjoint mode)

    Output parameters:
        Pooled node features of shape `(batch, n_node_features)`
        (if single mode, shape will be `(1, n_node_features)`).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_mean
        self.batch_pooling_op = tf.reduce_mean


def global_average_pool_layer_fabric(**kwargs):
    return GlobalAveragePool(**kwargs)
