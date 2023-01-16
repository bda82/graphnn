import logging

import numpy as np
import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.metrics import categorical_accuracy
from keras.optimizers import Adam

from gns.dataset.sfedu_dataset import sfedu_dataset_fabric
from gns.model.sfedu_conv_model import sfedu_model_fabric
from gns.loaders.disjoint_loader import DisjointLoader

logger = logging.getLogger(__name__)

logger.info(
    "A test industry example for the custom dataset to find appropriate vacancy for person."
)

# Config

logger.info("Let's define the test parameters...")

# Learning rate

learning_rate = 1e-2

# Number of training epochs
epochs = 400

# Patience for early stopping

es_patience = 10

# Batch size

batch_size = 32

# Generate Dataset and slice it to few batches

logger.info("Load dataset...")

data = sfedu_dataset_fabric(1000, n_labels=1000)

# Train/valid/test split

logger.info("Create Train/valid/test data collections...")

idxs = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]

# Data loaders

logger.info("Define loaders...")

loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

# Build model

logger.info("Build model...")

model = sfedu_model_fabric(data=data)
optimizer = Adam(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy()


# Fit model

logger.info("Fit model...")


@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    """
    Function for detecting accuracy and losses
    """
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    """
    Function for evaluating results
    """
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            # Keep track of batch size
            len(target),
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])


# Load training model

logger.info("Load training model...")

epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []


for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        
        logger.info("Compute validation loss and accuracy...")

        val_loss, val_acc = evaluate(loader_va)
        logger.info(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, val_acc
            )
        )

        # Check if loss improved for early stopping
        
        logger.info("Check if loss improved for early stopping...")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

# Evaluate model

logger.info("Evaluate model...")

model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)

logger.info("Completed...")
logger.info("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))
