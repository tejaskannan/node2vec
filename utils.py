import networkx as nx
import numpy as np
import json
import csv
from constants import *
from sklearn.cluster import KMeans


def create_mini_batches(walks, points, neg_samples, batch_size):
    # Randomly shuffle data points
    combined = list(zip(walks, points, neg_samples))
    np.random.shuffle(combined)
    walks, points, neg_samples = zip(*combined)

    walks_batches = []
    points_batches = []
    neg_sample_batches = []
    for i in range(0, len(walks), batch_size):
        walks_batches.append(np.array(walks[i:i+batch_size]))
        points_batches.append(np.array(points[i:i+batch_size]))
        neg_sample_batches.append(np.array(neg_samples[i:i+batch_size]))
    return {
        WALKS: np.array(walks_batches),
        POINTS: np.array(points_batches),
        NEG_SAMPLES: np.array(neg_sample_batches)
    }


def create_walk_windows(walks, window_size):
    walk_windows = []
    point_windows = []
    for walk in walks:
        for i in range(len(walk) - window_size):
            window = walk[i:i+window_size+1]
            walk_windows.append(window[1:])
            point_windows.append(window[0])
    return {
        WALKS: np.array(walk_windows),
        POINTS: np.array(point_windows)
    }


def create_negative_samples(points, walk_windows, point_windows, num_neg_samples, graph, use_edges):
    if use_edges:
        index = edge_index(graph)
        points = [index[p] for p in points]

    neg_samples = []
    for n, walk_window in zip(point_windows, walk_windows):
        w = set(walk_window)
        neg_sample = set()

        # Create negative samples outside of the current context
        while len(neg_sample) < num_neg_samples:
            u = np.random.choice(points)
            if (u != n) and (not u in w):
                neg_sample.add(u)

        neg_samples.append(list(neg_sample))
    return np.array(neg_samples)


def edge_index(graph):
    return {edge: i for i, edge in enumerate(graph.edges())}


def load_params(params_file_path):
    with open(params_file_path, 'r') as params_file:
        params = json.loads(params_file.read())
    return params


def cluster_edges(graph, edge_embeddings, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=1000).fit(edge_embeddings)
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


def append_lines_to_file(lines, file_path):
    with open(file_path, 'a') as file:
        for line in lines:
            file.write(line.strip() + '\n')


def load_data_file(file_path, graph, use_edges=False):
    if use_edges:
        index = edge_index(graph)

    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            elems = line.split()
            if use_edges:
                if len(elems) > 1:
                    data = [e.split('-') for e in elems]
                    data = [index[(int(e[0]), int(e[1]))] for e in data]
                else:
                    tokens = elems[0].split('-')
                    data = index[(int(tokens[0]), int(tokens[1]))]
            else:
                if len(elems) > 1:
                    data = [int(n) for n in elems]
                else:
                    data = int(elems[0])
            dataset.append(data)
    return dataset


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


def avg_2d_array(arr):
    s = 0.0
    count = 0.0
    for a in arr:
        s += np.sum(a)
        count += len(a)
    return s / count


def lst_elems_to_str(lst):
    return [str(x) for x in lst]


def lst_pairs_to_str(lst):
    return [str(x[0]) + '-' + str(x[1]) for x in lst]


def neg_softmax(edge_dict):
    arr = [-dist for dist in edge_dict.values()]
    max_elem = np.max(arr)
    exp_arr = np.exp(arr - max_elem)
    return exp_arr / np.sum(exp_arr)
