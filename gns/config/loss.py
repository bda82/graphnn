from dataclasses import dataclass


@dataclass
class Loss:
    categorical_crossentropy = "categorical_crossentropy"


def loss_fabric():
    return Loss()
