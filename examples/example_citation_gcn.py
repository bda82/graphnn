import logging

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping  # noqa
from tensorflow.keras.losses import CategoricalCrossentropy  # noqa
from tensorflow.keras.optimizers import Adam  # noqa

from gns.config.settings import settings_fabric
from gns.dataset.citation import citation_fabric
from gns.layer.gcn_convolution import GCNConvolutionalGeneralLayer
from gns.loaders.single_loader import single_loader_fabric
from gns.model.gcn import graph_convolutional_network_model_fabric
from gns.transformation.layer_process import layer_process_fabric
from gns.utils.mask_to_float_weights import mask_to_float_weights

settings = settings_fabric()

logger = logging.getLogger(__name__)

logger.info(
    "A test example for the Cora dataset (analysis of the citation graph of social network messages)."
)
logger.info("We will use Semi-Supervised Classification for ultra-precise HNN.")

logger.info("Let's define the test parameters...")

learning_rate = 1e-2
seed = 0
epochs = 200
patience = 10
data = "cora"

logger.info(
    "Initialize the initial values for the random value generator (scatterer)..."
)
tf.random.set_seed(seed=seed)

logger.info("Download dataset...")

dataset = citation_fabric(
    data,
    normalize_x=True,
    transforms=[layer_process_fabric(GCNConvolutionalGeneralLayer)],
)

logger.info("Define weights...")

weights_tr, weights_va, weights_te = (
    mask_to_float_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)

logger.info("Define model...")

model = graph_convolutional_network_model_fabric(n_labels=dataset.n_labels)
model.compile(
    optimizer=Adam(learning_rate),
    loss=CategoricalCrossentropy(reduction=settings.aggregation_methods.sum),
    weighted_metrics=["acc"],
)

logger.info("Train model...")

loader_tr = single_loader_fabric(dataset, sample_weights=weights_tr)
loader_va = single_loader_fabric(dataset, sample_weights=weights_va)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)
model.summary()

logger.info("Run model.")

loader_te = single_loader_fabric(dataset, sample_weights=weights_te)

eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)

logger.info("Completed...")
logger.info("Loss: {}\n" "Test accuracy: {}".format(*eval_results))
