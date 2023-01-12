import logging

logger = logging.getLogger(__name__)


class BellmanFordModifiedGraph:

    def __init__(self, vertices):
        self.V = vertices   # Total number of vertices in the graph
        self.graph = []     # Array of edges

    def add_edge(self, s, d, w, con, orien):
        """
        Add edges

        Args:
            s: graph node
            d: graph node
            w: edge weight
            con: connection type from list
            orin: edge orientation

        Returns:

        """
        self.graph.append([s, d, w, con, orien])

    def print_solution(self, dist):
        """
        Print solution.
        """
        logger.info("Vertex Distance from Source")
        for i in range(self.V):
            logger.info(f"{i}\t\t{dist[i]}")

    def build(self, src, flag1, searchVertex = None):
        """
        Main function.
        
        Args:
            src: start node
            flag1: First search criteria `o/tpvi` or  Second search criteria `0/1`
            searchVertex: searchVertex by default, shutdown, the second search criterion by type of connection, namely 'o', 'vi', 'tpvi'
        """
        if flag1 == 'o':
            # Step 1: fill the distance array and predecessor array

            dist = [float("Inf")] * self.V
            
            # Mark the source vertex
            
            dist[src] = 0

            # Step 2: relax edges |V| - 1 times
            
            minR = self.graph[0][2]
            for _ in range(self.V - 1):
                for s, d, w, con, orien in self.graph:
                    if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                        dist[d] = dist[s] + w
                        if dist[d] <= minR:
                            minR = dist[d]

            # Step 3: detect negative cycle
            # if value changes then we have a negative cycle in the graph
            # and we cannot find the shortest distances

            for s, d, w, con, orien in self.graph:
                if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                    logger.info("Graph contains negative weight cycle")
                    return

            self.print_solution(dist)

            logger.info(f"Nodes number: {self.V - 1}")
            logger.info(f"Vertex eccentricity: {dist[d]}")
            
            maxD = 0
            
            for g in self.graph:
                if g[2] > maxD:
                    maxD = g[2]
            
            logger.info(f"Graph diameter: {maxD}")
            logger.info(f"Graph radius: {minR}")
            
            minC = []
            for gr in self.graph:
                if gr[2] == minR:
                    minC.append(gr)

            logger.info(f"Graph center: {minC}")
        elif flag1 == 'tpvi':
            cache = []
            for par in self.graph:
                if par[3] == searchVertex:
                    cache.append(par)
                elif par[3] == searchVertex:
                    cache.append(par)
                elif par[3] == searchVertex:
                    cache.append(par)

            self.graph.clear()
            
            for g in cache:
                self.graph.append(g)
            
            logger.info(self.graph)
            
            # Step 1: fill the distance array and predecessor array
            
            dist = [float("Inf")] * self.V
            
            # Mark the source vertex
            
            dist[src] = 0

            # Step 2: relax edges |V| - 1 times
            minR = self.graph[0][2]
            for _ in range(self.V - 1):
                for s, d, w, con, orien in self.graph:
                    if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                        dist[d] = dist[s] + w
                        if dist[d] <= minR:
                            minR = dist[d]

            # Step 3: detect negative cycle
            # if value changes then we have a negative cycle in the graph
            # and we cannot find the shortest distances

            for s, d, w, con, orien in self.graph:
                if dist[s] != float("Inf") and dist[s] + w < dist[d]:
                    logger.info("Graph contains negative weight cycle")
                    return
            
            self.print_solution(dist)
            
            logger.info(f"Number of nodes: {self.V - 1}")
            logger.info(f"Vertex eccentricity: {dist[d]}")
            
            maxD = 0
            
            for g in self.graph:
                if g[2] > maxD:
                    maxD = g[2]

            logger.info(f"Graph diameter: {maxD}")
            logger.info(f"Graph radius: {minR}")

            minC = []
            
            for gr in self.graph:
                if gr[2] == minR:
                    minC.append(gr)
            logger.info('Центр графа : ', minC)


def bellman_ford_modified_fabric(vertices):
    return BellmanFordModifiedGraph(vertices)
