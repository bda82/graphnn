from dataclasses import dataclass


@dataclass
class AttributeProperties:
    initializer = "_initializer"
    regularizer = "_regularizer"
    constraint = "_constraint"
    activation = "activation"
    use_bias = "use_bias"
    dtype = "dtype"
    tocsr = "tocsr"
    toarray = "toarray"
    preprocess = "preprocess"


def attribute_properties_fabric():
    return AttributeProperties()
