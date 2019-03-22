import networkx as nx
import numpy as np
import json
import csv
from constants import *
from sklearn.cluster import KMeans


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


def append_to_log(row, log_path):
    with open(log_path, 'a') as log_file:
        csv_writer = csv.writer(log_file, delimiter=',', quotechar='|')
        csv_writer.writerow(row)
