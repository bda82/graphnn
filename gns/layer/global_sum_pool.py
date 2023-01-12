import tensorflow as tf

from gns.layer.global_pool import GlobalPoolLayer


class GlobalSumPoolLayer(GlobalPoolLayer):
    """
    Global Sum is an implementation of the GlobalPoolLayer base class.

    Combines a graph by calculating the sum of its node characteristics.

    Operating modes:
        single
        disjoint
        mixed
        batch

    Input parameters:
        Features of shape nodes `([batch], n_nodes, n_node_features)`
        Shape graph identifiers `(n_nodes, )` (works only for `disjoint` mode)

    Output parameters:
        Combined functions of the form node `(batch, n_node_features)`
            (if we work in single mode, the form should be `(1, n_node_features))`
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling_op = tf.math.segment_sum
        self.batch_pooling_op = tf.reduce_sum


def global_sum_layer_fabric(**kwargs):
    return GlobalPoolLayer(**kwargs)
