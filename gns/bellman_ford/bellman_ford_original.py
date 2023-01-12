import logging

logger = logging.getLogger(__name__)


class BellmanFordGraph:
    def __init__(self, v: int):
        self._v = v
        self._graph = []

    def append_edge(self, u, v, w) -> None:
        self._graph.append([u, v, w])

    def repr_attr(self, dist: list) -> None:
        logger.info("Let's determine the distance to the vertices from the source")
        for v in range(self._v):
            logger.info(f"{v}\t\t{dist[v]}")

    def build(self, src) -> None:
        """
        The main function that finds the shortest distance from the source
        to all other vertices using the Bellman-Ford algorithm.
        The function also defines a cycle with negative weights.
        """
        # Step 1: Initialize the distances from the source to other vertices as Infinity
        dist = [float("Inf")] * self._v
        dist[src] = 0

        # Step 2: Loosen all edges (|V|-1) times. The shortest path from the source
        # to any other vertex can pass through no more than (|V| - 1) edges
        for _ in range(self._v - 1):
            # Updating the distance value and parent index of neighboring vertices of the selected vertex.
            # Only those vertices that are still in the queue will be considered
            for u, v, w in self._graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        # Step 3: Check for negative weight cycles. Previous step
        # guarantees shortest distances if the graph does not contain
        # negative weight cycle. If we choose a shorter path, then this cycle will be there.

        for u, v, w in self._graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                logger.info("The graph contains a negative weight cycle")
                return

        self.repr_attr(dist)


def bellman_ford_orig_fabric(v: int):
    return BellmanFordGraph(v)
