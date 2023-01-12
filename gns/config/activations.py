from dataclasses import dataclass


@dataclass
class Activations:
    relu = "relu"
    softmax = "softmax"


def activations_fabric():
    return Activations()
