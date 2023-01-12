from dataclasses import dataclass


@dataclass
class Intializers:
    glorot_uniform = "glorot_uniform"
    zeros = "zeros"


def initializers_fabric():
    return Intializers()
