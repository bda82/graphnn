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

        os.makedirs(self.path, exist_ok=True)

        # Initialize the ArangoDB client.

        client = ArangoClient()

        # Connect to "_system" database as root user.

        db = client.db('_system', username='root', password='root_pass')

        # Get list of users

        user_collection = db.collection('users')
        users = user_collection.all()  # probably better to use .ids()
        users = np.fromiter(users, dtype=np.dtype(dict))
        users = sorted(users, key=lambda x: int(x['id']))

        # Get list of competencies

        competencies_collection = db.collection('competencies')
        competencies = competencies_collection.all()
        competencies = np.fromiter(competencies, dtype=np.dtype(dict))
        competencies = sorted(competencies, key=lambda x: int(x['id']))
        competencies_count = len(competencies)

        # User is an X axis
        for user in users:
            user_external_id = user['id']
            user_graph = 'user' + str(user_external_id)
            user_id = user["_id"]
            user_name = user["name"]
            print(">>>>", user_graph, user_id, user_name)

            # Build X axis

            x = np.zeros((competencies_count + 1, 1))  # competencies count plus 1 user (root of graph)

            x[0] = 1  # zero element is user and it's equal 1

            # Build A axis filled by zeros

            a = np.zeros((competencies_count + 1, competencies_count + 1))

            # Competency is an A axis
            for competence_from in competencies:
                comp_from_id = competence_from["_id"]
                comp_from_ext_id = competence_from["id"]
                comp_from_name = competence_from["name"]

                # If vacancy connected to competition directly
                # Build SQL-like query

                query = f'FOR v, e\n' \
                        f'  IN ANY SHORTEST_PATH\n' \
                        f'  "{user_id}" TO "{comp_from_id}"\n' \
                        f'  GRAPH "{user_graph}"\n' \
                        "  OPTIONS {weightAttribute:'distance'}\n" \
                        "  RETURN v._key"

                # If not empty, then path is exists

                v_connected = db.aql.execute(query)
                v_count = sum(1 for _ in v_connected)

                # Check for a non-zero count
                if v_count == 2:
                    # Skip duplicates
                    if a[0][int(comp_from_ext_id) - 1] == 0:
                        print("|", user_name, '->', comp_from_name)
                        a[int(comp_from_ext_id) - 1][0] = 1

                # Detect if competency connected to another one

                for comp_to in competencies:
                    comp_to_id = comp_to["_id"]
                    comp_to_ext_id = comp_to["id"]
                    comp_to_name = comp_to["name"]

                    # Skip if same
                    if comp_from_id == comp_to_id:
                        continue

                    # Build SQL-like query
                    query = f'FOR v, e\n' \
                            f'  IN ANY SHORTEST_PATH\n' \
                            f'  "{comp_from_id}" TO "{comp_to_id}"\n' \
                            f'  GRAPH "{user_graph}"\n' \
                            "  OPTIONS {weightAttribute:'distance'}\n" \
                            "  RETURN v._key"

                    # If not empty, then path is exists

                    v_connected = db.aql.execute(query)
                    v_count = sum(1 for _ in v_connected)

                    # Check for a non-zero count

                    if v_count == 2:
                        # Skip duplicates
                        if a[int(comp_to_ext_id) - 1][int(comp_from_ext_id) - 1] == 0:
                            print("|", user_name, comp_from_name, "->", comp_to_name)
                            a[int(comp_from_ext_id) - 1][int(comp_to_ext_id) - 1] = 1

            # Save graph of user

            filename = os.path.join(self.path, f'user_{user_external_id}')
            np.savez(
                filename,
                x=x,
                a=a,
            )

    def read(self):
        """Read data file"""
        mode = None
        if self.mode == 'u4v':
            mode = "user_*.npz"
        if self.mode == 'v4u':
            mode = "vacancy_*.npz"
        if mode is None:
            raise Exception('Mode is not supported')

        # Read files from disk

        _graphs = glob.glob(self.path + "/" + mode)

        # Output array

        graphs = []

        # Read graphs

        for graph in _graphs:
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
