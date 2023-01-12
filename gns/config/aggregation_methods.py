from dataclasses import dataclass


@dataclass
class AggregationMethods:
    sum = "sum"
    mean = "mean"
    max = "max"
    min = "min"
    prod = "prod"


def aggregation_methods_fabric():
    return AggregationMethods()
