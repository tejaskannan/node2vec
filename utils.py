import networkx as nx
import numpy as np
import json
from constants import *
from sklearn.cluster import KMeans


def create_random_walks(graph, num_walks, length, p, q):
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
                weight = 1.0 / (graph[v][x]['free_flow_time'] + SMALL_NUMBER)
                if x == t:
                    weight *= (1.0 / p)
                elif t in graph[x] or x in graph[t]:
                    weight *= 1.0
                else:
                    weight *= (1.0 / q)
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


def create_nodes_tensor(graph):
    return np.array(graph.nodes())


def create_mini_batches(walks, nodes, batch_size):
    walks_batches = []
    nodes_batches = []
    for walk in walks:
        # Randomly shuffle data points
        combined = list(zip(walk, nodes))
        np.random.shuffle(combined)
        w, n = zip(*combined)

        for i in range(0, len(w), batch_size):
            walks_batches.append(np.array(w[i:i+batch_size]))
            nodes_batches.append(np.array(n[i:i+batch_size]))
    return {
        WALKS: walks_batches,
        NODES: nodes_batches
    }


def load_params(params_file_path):
    with open(params_file_path, 'r') as params_file:
        params = json.loads(params_file.read())
    return params


def cluster_edges(graph, edge_embeddings, num_clusters):
    edge_emb_vectors = []
    for i in range(edge_embeddings.shape[0]):
        for j in range(edge_embeddings.shape[1]):
            if i in graph and j in graph[i]:
                vector = edge_embeddings[i, j]
                edge_emb_vectors.append(vector)
    edge_emb_vectors = np.array(edge_emb_vectors)

    kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=1000).fit(edge_emb_vectors)
    edge_labels = kmeans.labels_

    graph = graph.copy()
    for label, (u, v) in zip(edge_labels, graph.edges()):
        graph.add_edge(u, v, cluster=int(label))
    return graph


def cluster_nodes(graph, node_embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=1000).fit(node_embeddings)
    node_labels = kmeans.labels_
    graph = graph.copy()
    for label, node in zip(node_labels, graph.nodes()):
        graph.add_node(node, cluster=int(label))
    return graph
