import networkx as nx
import numpy as np
from constants import *


class RandomWalkerFactory:

    def get_random_walker(name, options):
        if name == 'node2vec':
            assert 'p' in options and 'q' in options, 'Must specify both p and q.'
            return Node2VecWalker(p=options['p'], q=options['q'])
        return None


class RandomWalker:

    def generate_walks(self, graph, num_walks, walk_length):
        raise NotImplementedError()


class Node2VecWalker(RandomWalker):

    def __init__(self, p, q):
        super(Node2VecWalker, self).__init__()
        self.p = p
        self.q = q

    def generate_walks(self, graph, num_walks, walk_length):
        walks = []
        for node in graph.nodes():

            t = node
            v = node
            walk = []

            while len(walk) < walk_length:

                walk.append(v)

                # Compute weights for each neighbor x of v
                neighbor_weights = []
                for x in graph.neighbors(v):
                    weight = 1.0 / (graph[v][x]['free_flow_time'] + SMALL_NUMBER)
                    if x == t:
                        weight *= (1.0 / self.p)
                    elif t in graph[x] or x in graph[t]:
                        weight *= 1.0
                    else:
                        weight *= (1.0 / self.q)
                    neighbor_weights.append(weight)

                # Normalize the weights
                neighbor_weights = np.array(neighbor_weights) / (np.sum(neighbor_weights) + SMALL_NUMBER)

                # Move the previous pointer to the current node after the first iteration
                if len(walk) > 0:
                    t = v

                # Select the next node
                if abs(1.0 - np.sum(neighbor_weights)) < 1e-3:
                    v = np.random.choice(list(graph.neighbors(v)), p=neighbor_weights)

            walks.append(walk)
        return np.array(walks)
