from dataclasses import dataclass


@dataclass
class Folders:
    citation = "Citation"
    project_main_folder = "gns"
    datasets = "datasets"
    models = "models"


def folders_fabric():
    return Folders()
