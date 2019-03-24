import networkx as nx
import numpy as np
import json
import csv
from constants import *
from sklearn.cluster import KMeans


def create_mini_batches(walks, nodes, neg_samples, batch_size):
    # Randomly shuffle data points
    combined = list(zip(walks, nodes, neg_samples))
    np.random.shuffle(combined)
    walks, nodes, neg_samples = zip(*combined)

    walks_batches = []
    nodes_batches = []
    neg_sample_batches = []
    for i in range(0, len(walks), batch_size):
        walks_batches.append(np.array(walks[i:i+batch_size]))
        nodes_batches.append(np.array(nodes[i:i+batch_size]))
        neg_sample_batches.append(np.array(neg_samples[i:i+batch_size]))
    return {
        WALKS: np.array(walks_batches),
        NODES: np.array(nodes_batches),
        NEG_SAMPLES: np.array(neg_sample_batches)
    }


def create_walk_windows(walks, window_size):
    walk_windows = []
    node_windows = []
    for walk in walks:
        for i in range(len(walk) - window_size - 1):
            window = walk[i:i+window_size+1]
            walk_windows.append(window[1:])
            node_windows.append(window[0])
    return {
        WALKS: np.array(walk_windows),
        NODES: np.array(node_windows)
    }


def create_negative_samples(nodes, walk_windows, node_windows, num_neg_samples):
    neg_samples = []
    for n, walk_window in zip(node_windows, walk_windows):
        w = set(walk_window)
        neg_sample = set()

        # Create negative samples outside of the current context
        while len(neg_sample) < num_neg_samples:
            u = np.random.choice(nodes)
            if (u != n) and (not u in w):
                neg_sample.add(u)

        neg_samples.append(list(neg_sample))
    return np.array(neg_samples)


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


def dist(a, b):
    degree_dist = (float(max(a[0], b[0])) / (min(a[0], b[0]) + SMALL_NUMBER)) - 1
    return degree_dist * max(a[1], b[1])


def compressed_degree_list(graph, nodes):
    degree_dict = {}
    for node in nodes:
        deg = graph.degree(node)
        if not deg in degree_dict:
            degree_dict[deg] = 0
        degree_dict[deg] += 1
    return sorted([(d, count) for d, count in degree_dict.items()], key=lambda t: t[0])
