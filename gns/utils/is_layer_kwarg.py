from gns.config.settings import settings_fabric
from gns.utils.is_keras_kwarg import KERAS_KWARGS

settings = settings_fabric()


LAYER_KWARGS = {"activation", "use_bias"}


def is_layer_kwarg(key):
    return key not in KERAS_KWARGS and (
        key.endswith(settings.attribute_properties.initializer)
        or key.endswith(settings.attribute_properties.regularizer)
        or key.endswith(settings.attribute_properties.constraint)
        or key in LAYER_KWARGS
    )
