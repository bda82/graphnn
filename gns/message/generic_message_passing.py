import inspect

import tensorflow as tf
from tensorflow.keras import backend as KerasBackend  # noqa
from tensorflow.keras.layers import Layer  # noqa

from gns.utils.deserialize_kwarg import deserialize_kwarg
from gns.utils.deserialize_scatter import deserialize_scatter
from gns.utils.is_keras_kwarg import is_keras_kwarg
from gns.utils.is_layer_kwarg import is_layer_kwarg
from gns.utils.serialize_kwarg import serialize_kwarg
from gns.utils.serialize_scatter import serialize_scatter


class GenericMessagePassing(Layer):
    """
    A base class for transmitting messages in a graph neural network.

    Works with models:
        single
        disjoint

    Notes:
        This layer and all its extensions expect a sparse adjacency matrix.

    By extending this class, you can create any level of message transmission in single/disjoint mode:
        single
        disjoint

    Distribution function distribute().
    ``sh
        distribute(x, a, e=no, **quargs)
    ```
    
    Distributes messages and calculates attachments for each node in the graph.

    Any additional parameter `kwargs` will be passed as an argument for the functions:
        `message()`
        `aggregate()`
        `update()`

    Message transmission function `message()`.
    ``sh
        message(x, **kwargs)
    ``
    
    Computes messages equivalent to (phi) in the theory of the method definition.
    Any additional parameter `kwargs` will be passed as an argument to
    the `propagate()` function if an attribute key match is found.

    The get_sources and get_targets functions are utility methods of the Layer ancestor class and will be used automatically,
    extracting attributes of nodes that send (sources) or receive (targets) a message.
    If you need direct access to the boundary indexes, then we can use the index_sources and index_targets attributes.

    `aggregate()` aggregation function.
    ```py
    aggregate(messages, **kwargs)
    ```
    
    Aggregates messages equivalent to (square) in the theory of the method definition.

    The behavior of this function can also be controlled using a keyword in the layer constructor (aggregate)
    Supported aggregations:
        sum
        mean
        max
        min
        prod

    Any additional keyword argument of this function will be filled with `propagate()` function
    if there is the corresponding keyword was found.

    Update function update().
    ```py
    update(embeddings, **kwargs)
    ```

    Updates aggregated messages to get the final node attachments equivalent to `gamma` parameter
    in the model definition.

    Any additional keyword argument of this function will also be filled with `propagate()` function
    if there is the corresponding keyword was found.

    Input parameters:
        `aggregate`: str or link to function, aggregation function.
        `kwargs`: additional class arguments compatible with the Keras layer, such as regulators, initializers, limiters, and so on
    
    This flag can be used to control the behavior of the `aggregate()` function without overriding it.

    Supported aggregations:
        sum
        mean
        max
        min
        prod

    If this parameter is passed as a reference to a function, this function must have the appropriate call signature
    for example, foo(updates, indices, n_nodes) and return a rank 2 tensor with the form (n_nodes, ...)

    """

    def __init__(self, aggregate="sum", **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})

        self.kwargs_keys = []

        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)

        self.msg_signature = inspect.signature(self.message).parameters
        self.agg_signature = inspect.signature(self.aggregate).parameters
        self.upd_signature = inspect.signature(self.update).parameters

        self.agg = deserialize_scatter(aggregate)

        self.built = False

    def call(self, inputs, **kwargs):
        """
        Call message passing.

        Args:

        Returns:

        """
        x, a, e = self.get_inputs(inputs)

        return self.propagate(x, a, e)

    def build(self, input_shape):
        """
        Build layer.

        Args:
            input_shape: input shape

        Returns:

        """
        self.built = True

    def propagate(self, x, a, e=None, **kwargs):
        """
        Propagate messages.

        Args:
            x: node features
            a: adjacency matrix
            e: edge features
            kwargs: additional attributes
        
        Returns:

        """
        self.n_nodes = tf.shape(x)[-2]

        # Вершины, которые получают сообщения

        self.index_targets = a.indices[:, 1]

        # Вершины, которые отправляют сообщения (например, соседние)

        self.index_sources = a.indices[:, 0]

        # Сообщения
        
        messages_kwargs = self.get_kwargs(x, a, e, self.msg_signature, kwargs)

        messages = self.message(x, **messages_kwargs)

        # Агрегация

        aggregated_kwargs = self.get_kwargs(x, a, e, self.agg_signature, kwargs)

        embeddings = self.aggregate(messages, **aggregated_kwargs)

        # Обновление

        updated_kwargs = self.get_kwargs(x, a, e, self.upd_signature, kwargs)

        output = self.update(embeddings, **updated_kwargs)

        return output

    def message(self, x, **kwargs):
        """
        Messages and sources.

        Args:
            x: node features
            kwargs: additional attributes
        
        Returns:

        """
        return self.get_sources(x)

    def aggregate(self, messages, **kwargs):
        """
        Aggregation of messages.

        Args:
            messages: messages
            kwargs: additional attributes
        
        Returns:

        """
        return self.agg(messages, self.index_targets, self.n_nodes)

    def update(self, embeddings, **kwargs):
        """
        Update messages.

        Args:
            embeddings: embeddings
            kwargs: additional attributes
        
        Returns:

        """
        return embeddings

    def get_targets(self, x):
        """
        Collect message targets.

        Args:
            x: node features

        Returns:

        """
        return tf.gather(x, self.index_targets, axis=-2)

    def get_sources(self, x):
        """
        Getting of message sources.

        Args:
            x: node features

        Returns:

        """
        return tf.gather(x, self.index_sources, axis=-2)

    def get_kwargs(self, x, a, e, signature, kwargs) -> dict:
        """
        Process attributes.

        Args:
            x: node features
            a: adjacency matrix
            e: edge features
            signature: signatures
            kwargs: additional attributes

        Returns:

        """
        output = {}

        for k in signature.keys():
            if signature[k].default is inspect.Parameter.empty or k == "kwargs":
                pass
            elif k == "x":
                output[k] = x
            elif k == "a":
                output[k] = a
            elif k == "e":
                output[k] = e
            elif k in kwargs:
                output[k] = kwargs[k]
            else:
                raise ValueError("Missing key {} for signature {}".format(k, signature))

        return output

    @staticmethod
    def get_inputs(inputs):
        """
        Analyzes input lists and returns a tuple `(x, a, e)` with node elements,
        the adjacency matrix and edge elements. If the input data contains only `x` and `a`, then
        returns `e = None`.

        Args:
            inputs: inputs

        Returns:

        """
        if len(inputs) == 3:
            x, a, e = inputs
            assert KerasBackend.ndim(e) in (2, 3), "`E` must have rank 2 or 3"
        elif len(inputs) == 2:
            x, a = inputs
            e = None
        else:
            raise ValueError(
                f"A tensor with 2 or 3 inputs `(X, A, E)` is expected, a tensor with {len(inputs)} inputs is obtained."
            )

        assert KerasBackend.ndim(x) in (2, 3), "`X` must have rank 2 or 3"
        assert KerasBackend.is_sparse(a), "`A` must have type SparseTensor"
        assert KerasBackend.ndim(a) == 2, "`A` must have rank 2"

        return x, a, e

    def get_config(self) -> dict:
        """
        Map config with some instances.
        """
        mapped_config = {"aggregate": serialize_scatter(self.agg)}

        keras_config = {}

        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))

        base_config = super().get_config()

        return {
            **base_config,
            **keras_config,
            **mapped_config,
            **self.config
        }

    @property
    def config(self) -> dict:
        return {}

    @staticmethod
    def preprocess(a):
        """
        Just dummy preprocess - we can define it for class inheritance.
        """
        return a


def generic_message_passing(aggregate="sum", **kwargs):
    return GenericMessagePassing(aggregate, **kwargs)
