import os
import os.path as osp

from gns.config.settings import settings_fabric

settings = settings_fabric()


_models_folder = os.path.join(
    "~", settings.folders.project_main_folder, settings.folders.models
)

MODEL_FOLDER = osp.expanduser(_models_folder)
