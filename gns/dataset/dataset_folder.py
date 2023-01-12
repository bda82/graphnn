import os
import os.path as osp

from gns.config.settings import settings_fabric

settings = settings_fabric()


_dataset_folder = os.path.join(
    "~", settings.folders.project_main_folder, settings.folders.datasets
)

DATASET_FOLDER = osp.expanduser(_dataset_folder)
