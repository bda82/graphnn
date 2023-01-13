from gns.config.settings import settings_fabric

settings = settings_fabric()


class LayerPreprocess(object):
    """
    Implements the preprocessing function in the convolutional layer for the adjacency matrix

    Input parameters:
        layer_class: layer class from the class of a layer from `gns.layers`, 
                     or any layer that overrides the preprocess(adj) method.
    """

    def __init__(self, layer_class):
        self.layer_class = layer_class

    def __call__(self, graph):
        if graph.a is not None and hasattr(self.layer_class, settings.attribute_properties.preprocess):
            graph.a = self.layer_class.preprocess(graph.a)

        return graph


def layer_process_fabric(layer_class):
    return LayerPreprocess(layer_class)
