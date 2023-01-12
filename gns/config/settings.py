from dataclasses import dataclass

from gns.config.activations import activations_fabric
from gns.config.aggregation_methods import aggregation_methods_fabric
from gns.config.attribute_properties import attribute_properties_fabric
from gns.config.constants import constants_fabric
from gns.config.data_types import data_types_fabric
from gns.config.datasets import datasets_fabric
from gns.config.folders import folders_fabric
from gns.config.functions_names import functions_names_fabric
from gns.config.initializers import initializers_fabric
from gns.config.loss import loss_fabric
from gns.config.models import models_fabric
from gns.config.names import names_fabric
from gns.config.urls import urls_fabric


@dataclass
class Settings:
    urls = urls_fabric()
    constants = constants_fabric()
    data_types = data_types_fabric()
    names = names_fabric()
    folders = folders_fabric()
    datasets = datasets_fabric()
    initializers = initializers_fabric()
    models = models_fabric()
    aggregation_methods = aggregation_methods_fabric()
    functions_names = functions_names_fabric()
    attribute_properties = attribute_properties_fabric()
    activations = activations_fabric()
    loss = loss_fabric()


def settings_fabric():
    return Settings()
