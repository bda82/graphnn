import os
import logging

from tensorflow.keras.layers import Dense  # noqa
from tensorflow.keras.models import Model  # noqa

from gns.layer.global_average_pool import global_average_pool_layer_fabric
from gns.layer.gcs_convolution import gsn_convolutional_general_layer_fabric
from gns.model.model_folder import MODEL_FOLDER
from gns.config.settings import settings_fabric

settings = settings_fabric()

logger = logging.getLogger(__name__)


class SfeduModel(Model):
    """
    Custom model for Industry example - find candidates vacancies
    """

    def __init__(self, data):
        super().__init__()

        logger.info("Mount data...")

        self.data = data

        logger.info("Create convolutional layers...")

        self.conv1 = gsn_convolutional_general_layer_fabric(32, activation=settings.activations.relu)
        self.conv2 = gsn_convolutional_general_layer_fabric(32, activation=settings.activations.relu)
        self.conv3 = gsn_convolutional_general_layer_fabric(32, activation=settings.activations.relu)

        logger.info("Create pooling...")

        self.global_pool = global_average_pool_layer_fabric()

        logger.info("Create Dense layer...")

        self.dense = Dense(data.n_labels, activation=settings.activations.softmax)

    def call(self, inputs):
        """
        Call layer.

        Args:
            inputs: inputs
            mask: mask

        Returns:

        """
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        output = self.global_pool([x, i])
        output = self.dense(output)

        return output


def sfedu_model_fabric(data, **kwargs):
    return SfeduModel(data, **kwargs)  # noqa


def path():
    return os.path.join(MODEL_FOLDER, 'SfeduModel')
