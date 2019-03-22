import networkx as nx
import numpy as np
import json
from constants import *


def create_random_walks(graph, length, p, q):
    walks = []
    for node in graph.nodes():
        t = node
        v = node
        walk = []

        while len(walk) < length:

            walk.append(v)

            # Compute weights for each neighbor x of v
            neighbor_weights = []
            for x in graph.neighbors(v):
                weight = graph[v][x]['length']
                if x == t:
                    weight *= (1.0 / p)
                elif x in graph[t]:
                    weight *= 1
                else:
                    weight *= (1.0 / q)
                neighbor_weights.append(weight)

            # Normalize the weights
            neighbor_weights = np.array(neighbor_weights) / (np.sum(neighbor_weights) + SMALL_NUMBER)

            # Move the previous pointer to the current node after the first iteration
            if len(walk) > 0:
                t = v

            # Select the next node
            if (1.0 - np.sum(neighbor_weights)) < SMALL_NUMBER:
                v = np.random.choice(list(graph.neighbors(v)), p=neighbor_weights)

        walks.append(walk)
    return np.array(walks)


def create_nodes_tensor(graph):
    return np.array(graph.nodes())


def create_mini_batches(walks, nodes, batch_size):
    # Randomly shuffle data points
    combined = list(zip(walks, nodes))
    np.random.shuffle(combined)
    walks, nodes = zip(*combined)

    walks_batches = []
    nodes_batches = []
    for i in range(0, len(walks), batch_size):
        walks_batches.append(np.array(walks[i:i+batch_size]))
        nodes_batches.append(np.array(nodes[i:i+batch_size]))
    return {
        WALKS: walks_batches,
        NODES: nodes_batches
    }


def load_params(params_file_path):
    with open(params_file_path, 'r') as params_file:
        params = json.loads(params_file.read())
    return params