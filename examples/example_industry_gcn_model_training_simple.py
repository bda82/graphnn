import tensorflow as tf
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam

from gns.dataset.tech_dataset import tech_dataset_fabric
from gns.layer.gcn_convolution import GCNConvolutionalGeneralLayer
from gns.model.gcn import GraphConvolutionalNetworkModel
from gns.transformation.layer_process import LayerPreprocess
from gns.transformation.adj_to_sp_tensor import AdjToSpTensor
from gns.loaders.single_loader import single_loader_fabric

learning_rate = 1e-2  # Learning rate
epochs = 200  # Number of epochs for training
seed = 0  # Make weight initialization reproducible

tf.random.set_seed(seed=seed)

# Load data
dataset = tech_dataset_fabric(
    name='jd_data2',
    normalize_x=True,
    transforms=[
        LayerPreprocess(GCNConvolutionalGeneralLayer),
        AdjToSpTensor()
    ]
)
graph = dataset[0]
x, a, y = graph.x, graph.a, graph.y
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te

model = GraphConvolutionalNetworkModel(n_labels=dataset.n_labels, activation="softmax")
optimizer = Adam(learning_rate=learning_rate)
loss_fn = CategoricalCrossentropy()


# Training step
@tf.function
def train():
    with tf.GradientTape() as tape:
        predictions = model([x, a], training=True)
        loss = loss_fn(y[mask_tr], predictions[mask_tr])
        loss += sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Time the execution of 200 epochs of training
train()  # Warm up to ignore tracing times when timing
for epoch in range(1, epochs):
    loss = train()
    print(f"epoch = {epoch} / {epochs}, loss = {loss}")

print(f"Final loss = {loss}")
