import sys
import os
import numpy as np
from gns.graph.graph import Graph
from gns.dataset.dataset import Dataset
from arango import ArangoClient
import glob
import scipy.sparse as sp


class ArangoDataset(Dataset):
    def __init__(self, mode="u4v", **kwargs):
        self.mode = type
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.dirname(__file__) + '/ArangoDataset'

    def download(self):

        # Create the directory
        os.mkdir(self.path)

        # Initialize the ArangoDB client.
        client = ArangoClient()

        # Connect to "_system" database as root user.
        db = client.db('_system', username='root', password='root_pass')

        # Get list of users
        uCollection = db.collection('users')
        users = uCollection.all()  # probably better to use .ids()
        users = np.fromiter(users, dtype=np.dtype(dict))
        users = sorted(users, key=lambda x: int(x['id']))

        # Get list of competencies
        cCollection = db.collection('competencies')
        competencies = cCollection.all()
        competencies = np.fromiter(competencies, dtype=np.dtype(dict))
        competencies = sorted(competencies, key=lambda x: int(x['id']))
        cCount = len(competencies)

        # User is an X axis
        for user in users:
            userExtId = user['id']
            userGraph = 'user' + str(userExtId)
            userId = user["_id"]
            userName = user["name"]
            print(">>>>", userGraph, userId, userName)

            # Build X axis
            x = np.zeros((cCount + 1, 1))  # competencies count plus 1 user (root of graph)
            x[0] = 1  # zero element is user and it's equal 1

            # Build A axis filled by zeros
            a = np.zeros((cCount + 1, cCount + 1))

            # Competency is an A axis
            for compFrom in competencies:
                compFromId = compFrom["_id"]
                compFromExtId = compFrom["id"]
                compFromName = compFrom["name"]

                #
                # If vacancy connected to competition directly
                #

                # Build query
                query = f'FOR v, e\n' \
                        f'  IN ANY SHORTEST_PATH\n' \
                        f'  "{userId}" TO "{compFromId}"\n' \
                        f'  GRAPH "{userGraph}"\n' \
                        "  OPTIONS {weightAttribute:'distance'}\n" \
                        "  RETURN v._key"

                # If not empty, then path is exists
                vConnected = db.aql.execute(query)
                vCount = sum(1 for _ in vConnected)

                # Check for a non-zero count
                if vCount == 2:
                    # Skip duplicates
                    if a[0][int(compFromExtId) - 1] == 0:
                        print("|", userName, '->', compFromName)
                        a[int(compFromExtId) - 1][0] = 1

                # Detect if competency connected to another one
                for compTo in competencies:
                    compToId = compTo["_id"]
                    compToExtId = compTo["id"]
                    compToName = compTo["name"]

                    # Skip if same
                    if compFromId == compToId:
                        continue

                    # Build query
                    query = f'FOR v, e\n' \
                            f'  IN ANY SHORTEST_PATH\n' \
                            f'  "{compFromId}" TO "{compToId}"\n' \
                            f'  GRAPH "{userGraph}"\n' \
                            "  OPTIONS {weightAttribute:'distance'}\n" \
                            "  RETURN v._key"

                    # If not empty, then path is exists
                    vConnected = db.aql.execute(query)
                    vCount = sum(1 for _ in vConnected)

                    # Check for a non-zero count
                    if vCount == 2:
                        # Skip duplicates
                        if a[int(compToExtId) - 1][int(compFromExtId) - 1] == 0:
                            print("|", userName, compFromName, "->", compToName)
                            a[int(compFromExtId) - 1][int(compToExtId) - 1] = 1

            # Save graph of user
            filename = os.path.join(self.path, f'user_{userExtId}')
            np.savez(
                filename,
                x=x,
                a=a,
            )

    def read(self):

        mode = None
        if self.mode == 'u4v':
            mode = "user_*.npz"
        if self.mode == 'v4u':
            mode = "vacancy_*.npz"
        if mode == None:
            raise Exception('Mode is not supported')

        # Read files from disk
        _graphs = glob.glob(self.path + "/" + mode)

        # Output array
        graphs = []

        # Read graphs
        for graph in _graphs:
            # Read from file
            data = np.load(graph, allow_pickle=True)

            # Process parameters
            x = data['x']
            a = data['a']
            # a = sp.csr_matrix(a)
            y = data['y']

            # Append arrays to Graph object
            graphs.append(Graph(x=x, a=a, y=y))

        # Return array of graphs
        return graphs


def arango_fabric(type="u4v", **kwargs):
    return ArangoDataset(type=type, **kwargs)
