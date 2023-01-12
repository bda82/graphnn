from gns.bellman_ford.bellman_ford_modified import bellman_ford_modified_fabric

# define key parameters

vertex_count = 5

# create BellmanFord instance

graph = bellman_ford_modified_fabric(vertex_count)

# append some edges to calculate distance

graph.add_edge(0, 1, 5, 'o', 0)
graph.add_edge(0, 2, 4, 'o', 1)
graph.add_edge(1, 3, 5, 'o', 1)
graph.add_edge(2, 1, 6, 'o', 0)
graph.add_edge(3, 2, 2, 'o', 0)

# build graph distances

graph.build(0, 'tpvi', 'o')