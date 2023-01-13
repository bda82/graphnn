from tensorflow.keras import (  # noqa
    activations,  # noqa
    constraints,  # noqa
    initializers,  # noqa
    regularizers,  # noqa
)

from gns.config.settings import settings_fabric

settings = settings_fabric()


def deserialize_kwarg(key, attr):
    """
    Attribute deserialization.

    Args:
        key: attribute key
        attr: attribute
    
    Returns:
        deserialized attribute
    """
    if key.endswith(settings.attribute_properties.initializer):
        return initializers.get(attr)

    if key.endswith(settings.attribute_properties.regularizer):
        return regularizers.get(attr)

    if key.endswith(settings.attribute_properties.constraint):
        return constraints.get(attr)

    if key == settings.attribute_properties.activation:
        return activations.get(attr)

    return attr
