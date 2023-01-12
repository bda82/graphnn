from tensorflow.keras import (  # noqa
    activations,
    constraints,
    initializers,
    regularizers,
)

from gns.config.settings import settings_fabric

settings = settings_fabric()


def serialize_kwarg(key, attr):
    """
    Serialize attributes.

    Args:
        key: ключ атрибута
        attr: атрибут
    
    Returns:
        serialized attribute
    """
    if key.endswith(settings.attribute_properties.initializer):
        return initializers.serialize(attr)

    if key.endswith(settings.attribute_properties.regularizer):
        return regularizers.serialize(attr)

    if key.endswith(settings.attribute_properties.constraint):
        return constraints.serialize(attr)

    if key == settings.attribute_properties.activation:
        return activations.serialize(attr)
        
    if key == settings.attribute_properties.use_bias:
        return attr
