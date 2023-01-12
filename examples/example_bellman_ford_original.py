from gns.bellman_ford.bellman_ford_original import bellman_ford_orig_fabric

# define key parameters

vertex_count = 5

# create BellmanFord instance

graph = bellman_ford_orig_fabric(vertex_count)

# append some edges to calculate distance

graph.append_edge(0, 1, -1)
graph.append_edge(0, 2, 4)
graph.append_edge(1, 2, 3)
graph.append_edge(1, 3, 2)
graph.append_edge(1, 4, 2)
graph.append_edge(3, 2, 5)
graph.append_edge(3, 1, 1)
graph.append_edge(4, 3, -3)

# build graph distance

graph.build(0)
