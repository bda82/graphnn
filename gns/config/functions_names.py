from dataclasses import dataclass


@dataclass
class FunctionsNames:
    sum = "sum"
    mean = "mean"
    max = "max"
    min = "min"
    prod = "prod"


def functions_names_fabric():
    return FunctionsNames()
