from dataclasses import dataclass


@dataclass
class Constants:
    citation_suffixes = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]


def constants_fabric():
    return Constants()
