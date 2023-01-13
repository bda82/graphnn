from dataclasses import dataclass

from gns.config.names import names_fabric

names = names_fabric()


@dataclass
class Datasets:
    available_datasets = [names.cora, names.citeseer, names.pubmed]
    dataset_prefix_index = "ind."


def datasets_fabric():
    return Datasets()
