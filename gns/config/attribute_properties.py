from dataclasses import dataclass


@dataclass
class AttributeProperties:
    initializer = "_initializer"
    regularizer = "_regularizer"
    constraint = "_constraint"
    activation = "activation"
    use_bias = "use_bias"


def attribute_properties_fabric():
    return AttributeProperties()
