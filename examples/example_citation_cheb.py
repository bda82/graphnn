import logging

from tensorflow.keras.callbacks import EarlyStopping  # noqa
from tensorflow.keras.layers import Dropout, Input  # noqa
from tensorflow.keras.losses import CategoricalCrossentropy  # noqa
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.optimizers import Adam  # noqa
from tensorflow.keras.regularizers import l2  # noqa

from gns.config.settings import settings_fabric
from gns.dataset.citation import citation_fabric
from gns.layer.cheb import ChebyshevConvolutionalLayer
from gns.loaders.single_loader import single_loader_fabric
from gns.transformation.layer_process import layer_process_fabric
from gns.utils.mask_to_weights import mask_to_simple_weights

settings = settings_fabric()

logger = logging.getLogger(__name__)

logger.info(
    "Test example for the Cora dataset for the Chebyshev Convolutional layer (analysis of the citation graph of social network messages)."
)
logger.info("We will use convolutional layer")

logger.info("Download dataset...")

dataset = citation_fabric(
    "cora", transforms=[layer_process_fabric(ChebyshevConvolutionalLayer)]
)

logger.info("Calculate weights...")

weights_tr, weights_va, weights_te = (
    mask_to_simple_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)

logger.info("Define parameters...")

# Number of channels for the first layer
channels = 16

# The maximum power of the Chebyshev polynomial
K = 2

# Dropout percentage for functions
dropout = 0.5

# Regularization parameter
l2_reg = 2.5e-4

# Traning ration
learning_rate = 1e-2

# Training epoches
epochs = 200

# Limit for early stopping of training
patience = 10

logger.info("Определим модель...")

# The number of graph nodes
N = dataset.n_nodes

# The original size of the node features
F = dataset.n_node_features

# Labels quantity
n_out = dataset.n_labels

a_dtype = dataset[0].a.dtype
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)

logger.info("Let's define the layers of a graph neural network...")

do_1 = Dropout(dropout)(x_in)
gc_1 = ChebyshevConvolutionalLayer(  # noqa
    channels,
    K=K,
    activation=settings.activations.relu,
    kernel_regularizer=l2(l2_reg),
    use_bias=False,
)([do_1, a_in])

do_2 = Dropout(dropout)(gc_1)
gc_2 = ChebyshevConvolutionalLayer(  # noqa
    n_out, K=K, activation=settings.activations.softmax, use_bias=False
)([do_2, a_in])


logger.info("Build model...")

model = Model(inputs=[x_in, a_in], outputs=gc_2)
model.compile(
    optimizer=Adam(lr=learning_rate),
    loss=CategoricalCrossentropy(
        reduction=settings.aggregation_methods.sum
    ),  # To calculate the average
    weighted_metrics=["acc"],
)
model.summary()

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

# Run model

logger.info("Run model...")

loader_te = single_loader_fabric(dataset, sample_weights=weights_te)

eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)

logger.info("Completed...")
logger.info("Loss: {}\n" "Test accuracy: {}".format(*eval_results))
