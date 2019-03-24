import networkx as nx
import numpy as np
from constants import *
from fastdtw import fastdtw
from utils import dist, compressed_degree_list, avg_2d_array


class RandomWalkerFactory:

    def get_random_walker(name, options):
        if name == 'node2vec':
            assert 'p' in options and 'q' in options, 'Must specify both p and q.'
            return Node2VecWalker(p=options['p'], q=options['q'])
        if name == 'struc2vec':
            assert 'q' in options
            k_max = options['k_max'] if 'k_max' in options else -1
            n_comparisons = options['n_comparisons'] if 'n_comparisons' in options else -1
            return Struc2VecWalker(q=options['q'], k_max=k_max, n_comp=n_comparisons)
        return None


class RandomWalker:

    def generate_walks(self, graph, walk_length, num_walks):
        raise NotImplementedError()


class Node2VecWalker(RandomWalker):

    def __init__(self, p, q):
        super(Node2VecWalker, self).__init__()
        self.p = p
        self.q = q

    def generate_walks(self, graph, walk_length, num_walks):
        walks = []
        for _ in range(num_walks):
            for node in graph.nodes():

                t = node
                v = node
                walk = []

                while len(walk) < walk_length:

                    walk.append(v)

                    # Compute weights for each neighbor x of v
                    neighbor_weights = []
                    for x in graph.neighbors(v):
                        weight = 1.0
                        if x == t:
                            weight *= (1.0 / self.p)
                        elif t in graph[x] or x in graph[t]:
                            weight *= 1.0
                        else:
                            weight *= (1.0 / self.q)
                        neighbor_weights.append(weight)

                    # Normalize the weights
                    s = np.sum(neighbor_weights) + SMALL_NUMBER
                    neighbor_weights = np.array(neighbor_weights) / s

                    # Move the previous pointer to the current node after the first iteration
                    if len(walk) > 0:
                        t = v

                    # Select the next node
                    if abs(1.0 - np.sum(neighbor_weights)) < 1e-3:
                        v = np.random.choice(list(graph.neighbors(v)), p=neighbor_weights)

                walks.append(walk)
        return np.array(walks)


class Struc2VecWalker(RandomWalker):

    def __init__(self, q, k_max, n_comp):
        super(Struc2VecWalker, self).__init__()
        self.q = q
        self.k_max = k_max
        self.num_comparisons = n_comp

    def generate_walks(self, graph, walk_length, num_walks):

        # Dictionary mapping each vertex to a list of vertices which have
        # similar degrees
        degree_clusters = self._create_degree_clusters(graph)

        # Vertices which are direct neighbors of each node
        neighborhoods = {
            node: {0: set([node])} for node in graph.nodes()
        }

        # Create degree lists for the 0th level
        degree_neighborhoods = {node: {} for node in graph.nodes()}
        self._add_kth_degree_neighborhood(graph, degree_neighborhoods, neighborhoods, 0)

        # Initialize 0th level distances and weights
        l0_dist = self._compute_distances(graph, degree_neighborhoods, degree_clusters, 0, None)
        distances = [l0_dist]

        l0_weights = self._compute_weights(graph, l0_dist)
        weights = [l0_weights]
        avg_weights = [avg_2d_array(l0_weights)]

        walks = []
        for _ in range(num_walks):
            for node in graph.nodes():
                walk = [node]
                k = 0
                u = node
                while len(walk) < walk_length:
                    should_stay = np.random.random() < self.q
                    if should_stay:
                        u = np.random.choice(degree_clusters[u], p=weights[k][u])
                        walk.append(u)
                    else:
                        if k == 0:
                            k += 1
                        elif self.k_max != -1 and k == self.k_max:
                            k -= 1
                        else:
                            gamma = np.sum([int(weights[k][u][v] > avg_weights[k]) \
                                            for v,_ in enumerate(degree_clusters[u])])
                            up_weight = np.log(gamma + np.e) 
                            down_weight = 1.0
                            up_prob = up_weight / (up_weight + down_weight)
                            should_move_up = np.random.random() < up_prob
                            if should_move_up:
                                k += 1
                            else:
                                k -= 1

                        # Only compute a layer's weights when the layer is reached
                        if len(weights) <= k:
                            self._add_kth_neighborhood(graph, neighborhoods, k)
                            self._add_kth_degree_neighborhood(graph, degree_neighborhoods, neighborhoods, k)
                            
                            lk_dist = self._compute_distances(graph, degree_neighborhoods,
                                                              degree_clusters, k, distances[k-1])
                            lk_weights = self._compute_weights(graph, lk_dist)

                            distances.append(lk_dist)
                            weights.append(lk_weights)
                            avg_weights.append(avg_2d_array(lk_weights))

                walks.append(walk)
        return np.array(walks)

    def _add_kth_neighborhood(self, graph, neighborhoods, k):
        for node in graph.nodes():
            prev = neighborhoods[node][k-1]
            neighbors = prev.copy()
            for n in prev:
                neighbors.update(set(graph.neighbors(n)))

            neighborhoods[node][k] = neighbors

    def _add_kth_degree_neighborhood(self, graph, degree_neighborhoods, neighborhoods, k):
        for node in graph.nodes():
            degree_neighborhoods[node][k] = compressed_degree_list(graph, neighborhoods[node][k])
        return degree_neighborhoods

    def _compute_distances(self, graph, degree_neighborhoods, degree_clusters, k, prev_layer_distances):
        distances = []
        for u in graph.nodes():
            d = [self._compute_distance(u, v, i, degree_neighborhoods, k, prev_layer_distances) \
                 for i, v in enumerate(degree_clusters[u])]
            distances.append(d)

        return np.array(distances)

    def _compute_distance(self, u, v, v_index, degree_neighborhoods, k, prev_layer_distances):
        if u == v:
            return 0.0
        distance, _ = fastdtw(degree_neighborhoods[u][k], degree_neighborhoods[v][k], dist=dist)
        f_k = prev_layer_distances[u][v_index] + distance if prev_layer_distances is not None else distance
        return f_k

    def _compute_weights(self, graph, distances):
        return [self._compute_weight(distances, n) for n in graph.nodes()]

    def _compute_weight(self, distances, u):
        distances = np.array(distances[u])
        weights = np.exp(-distances)
        s = np.sum(weights)
        weights = weights / s
        return np.array(weights)

    def _create_degree_clusters(self, graph):
        degrees = [graph.degree(n) for n in graph.nodes()]
        sorted_nodes = [n for _, n in sorted(zip(degrees, list(graph.nodes())))]
        degree_clusters = {}

        # Determine the size of cluster for each node
        cluster_size = self.num_comparisons
        if cluster_size == -1:
            cluster_size = int(np.log(graph.number_of_nodes())) + 1

        for i, n in enumerate(sorted_nodes):
            if i < cluster_size:
                cluster = sorted_nodes[0:cluster_size+i]
            elif i >= len(sorted_nodes) - cluster_size:
                cluster = sorted_nodes[i-cluster_size:]
            else:
                cluster = sorted_nodes[i-cluster_size:i+cluster_size]
            cluster.remove(n)
            degree_clusters[n] = cluster
        return degree_clusters
