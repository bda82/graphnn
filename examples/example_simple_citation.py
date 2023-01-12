import logging

from tensorflow.keras.callbacks import EarlyStopping  # noqa
from tensorflow.keras.layers import Input  # noqa
from tensorflow.keras.models import Model  # noqa
from tensorflow.keras.optimizers import Adam  # noqa
from tensorflow.keras.regularizers import l2  # noqa

from gns.config.settings import settings_fabric
from gns.dataset.citation import citation_fabric
from gns.layer.gcn_convolution import (
    GCNConvolutionalGeneralLayer,
    gcn_convolutional_general_layer_fabric,
)
from gns.loaders.single_loader import single_loader_fabric
from gns.transformation.layer_process import layer_process_fabric

settings = settings_fabric()

logger = logging.getLogger(__name__)

logger.info(
    "A simple test case for the Cora dataset (analysis of the citation graph of social network messages)."
)
logger.info(
    "We will use a simplified GNN model with a custom transformation for the adjacency matrix."
)


class CustomGCN:
    def __init__(self, K):
        self.K = K

    def __call__(self, graph):
        out = graph.a
        for _ in range(self.K - 1):
            out = out.dot(out)
        out.sort_indices()
        graph.a = out
        return graph


logger.info("Define parameters...")

K = 2  # Число этапов распространения для CustomGCN

logger.info("Download dataset...")

dataset = citation_fabric(
    "cora",
    transforms=[layer_process_fabric(GCNConvolutionalGeneralLayer), CustomGCN(K)],
)

mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

# Regularization parameter
l2_reg = 5e-6

# Learning rate
learning_rate = 0.2

# Number of epochs for training
epochs = 20000

# Limit for early stopping of training
patience = 200

# Number of graph nodes
N = dataset.n_nodes

# The original size of the node features
F = dataset.n_node_features

# Number of labels
n_out = dataset.n_labels

a_dtype = dataset[0].a.dtype
x_in = Input(shape=(F,))
a_in = Input((N,), sparse=True, dtype=a_dtype)

output = gcn_convolutional_general_layer_fabric(  # noqa
    n_out,
    activation=settings.activations.softmax,
    kernel_regularizer=l2(l2_reg),
    use_bias=False,
)([x_in, a_in])

logger.info("Let's assemble the model...")

model = Model(inputs=[x_in, a_in], outputs=output)
model.compile(
    optimizer=Adam(lr=learning_rate),
    loss=settings.loss.categorical_crossentropy,
    weighted_metrics=["acc"],
)
model.summary()

logger.info("Let's train the model...")

loader_tr = single_loader_fabric(dataset, sample_weights=mask_tr)
loader_va = single_loader_fabric(dataset, sample_weights=mask_va)
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)

# Let's start the model

logger.info("Let's start the model...")

loader_te = single_loader_fabric(dataset, sample_weights=mask_te)

eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)

logger.info("Completed...")
logger.info("Loss: {}\n" "Test accuracy: {}".format(*eval_results))
