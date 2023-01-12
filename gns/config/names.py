from dataclasses import dataclass


@dataclass
class Names:
    citeseer = "citeseer"
    cora = "cora"
    bias = "bias"
    pubmed = "pubmed"
    kernel = "kernel"
    kernel_1 = "kernel_1"
    kernel_2 = "kernel_2"


def names_fabric():
    return Names()
