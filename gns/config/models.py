from dataclasses import dataclass


@dataclass
class Models:
    single = "single"
    disjoint = "disjoint"
    mixed = "mixed"
    batch = "batch"


def models_fabric():
    return Models()
