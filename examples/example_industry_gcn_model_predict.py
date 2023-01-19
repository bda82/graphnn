import logging
import keras
import numpy as np
import os
import scipy.sparse as sp
from pathlib import Path
from arango import ArangoClient
from gns.config.settings import settings_fabric
from gns.layer.gcn_convolution import GCNConvolutionalGeneralLayer
from gns.model.gcn import GraphConvolutionalNetworkModel
from gns.model.gcn import path
from gns.dataset.dataset_folder import DATASET_FOLDER

settings = settings_fabric()
datasetsPath = DATASET_FOLDER + '/ArangoTechDataset/users'

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Load model...")
filepath = path() + '/example_industry_gcn_model'
model = keras.models.load_model(
    filepath,
    custom_objects={
        'GCNConvolutionalGeneralLayer': GCNConvolutionalGeneralLayer,
        'GraphConvolutionalNetworkModel': GraphConvolutionalNetworkModel
    },

)

logger.info("Define user graph...")

# Connect to database
client = ArangoClient()
db = client.db('_system', username='root', password='root_pass')

# Get all nodes (competencies)
nodes_collection = db.collection('nodes')
nodes = nodes_collection.all()
nodes = np.fromiter(nodes, dtype=np.dtype(dict))
nodes = sorted(nodes, key=lambda x: int(x['_key']))
nodes_count = len(nodes)


def make_graph(userGraph, userId, nodes):
    """
    Generate and save graph of user (if not exists),
    or just read ot and return.
    """

    # Return file if already exists
    file = datasetsPath + f'/{userGraph}'
    fileExt = datasetsPath + f'/{userGraph}.npz'
    file_path = Path(fileExt)
    if file_path.is_file():
        data = np.load(file_path, allow_pickle=True)
        return {'a': data['a'], 'x': data['x']}

    # By default, we have only one feature per each node
    n_features = 1

    # Build empty arrays
    x = np.zeros((nodes_count, n_features))  # Nodes and features
    a = np.zeros((nodes_count, nodes_count))  # Edges (matrix of connections between nodes)

    # a = (I, J)
    for idxI, nodeI in enumerate(nodes):
        #
        # Step 1 - Detect all nodes to which user connected directly
        #

        # Some variables
        nodeId = nodeI['_id']

        # If user connected to nodes directly
        query = f'FOR v, e\n' \
                f'  IN ANY SHORTEST_PATH\n' \
                f'  "{userId}" TO "{nodeId}"\n' \
                f'  GRAPH "{userGraph}"\n' \
                "  OPTIONS {weightAttribute:'distance'}\n" \
                "  RETURN v._id"

        # If not empty, then path is exists
        v_connected = db.aql.execute(query)
        v_count = sum(1 for _ in v_connected)

        # Check for a non-zero count
        if v_count == 2:
            # Make active nodes connected to user
            x[idxI][0] = 1
            # Nodes connected to user will be connected to himself
            a[idxI][idxI] = 1

        #
        # Step 2 - Detect all nodes connected to each other
        #
        for idxJ, nodeJ in enumerate(nodes):
            # Some variables
            nodeJd = nodeJ['_id']

            # If user connected to nodes directly
            query = f'FOR v, e\n' \
                    f'  IN ANY SHORTEST_PATH\n' \
                    f'  "{nodeId}" TO "{nodeJd}"\n' \
                    f'  GRAPH "{userGraph}"\n' \
                    "  OPTIONS {weightAttribute:'distance'}\n" \
                    "  RETURN v._id"

            # If not empty, then path is exists
            v_connected = db.aql.execute(query)
            v_count = sum(1 for _ in v_connected)

            # Check for a non-zero count
            if v_count == 2:
                # Make active nodes connected to user
                x[idxI][0] = 1
                # Nodes connected to user will be connected to himself
                a[idxJ][idxI] = 1

    # Save graph of user
    os.makedirs(datasetsPath, exist_ok=True)
    np.savez(file, x=x, a=a)

    # Return generated graph of user
    return {'a': a, 'x': x}


# Example user from Graphs list in ArangoDB
userGraph = 'user-v2-1'
userId = 'users/79608'

# Generate user matrixes
data = make_graph(userGraph=userGraph, userId=userId, nodes=nodes)
x = data['x']
x = x.astype(np.float32)
a = data['a']
a = a.astype(np.float32)

logger.info("Predict results...")
predictions = model([x, a], training=False)
print(predictions)
