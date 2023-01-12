from dataclasses import dataclass


@dataclass
class Folders:
    citation = "Citation"
    project_main_folder = "gns"
    datasets = "datasets"


def folders_fabric():
    return Folders()
