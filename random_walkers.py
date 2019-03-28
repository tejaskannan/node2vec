import networkx as nx
import numpy as np
from constants import *
from fastdtw import fastdtw
from utils import dist, compressed_degree_list, avg_2d_array
from utils import neg_softmax


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
        if name == 'edge2vec':
            assert 'q' in options
            field = options['field'] if 'field' in options else 'weight'
            return Edge2VecWalker(q=options['q'], field=field)
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


class Edge2VecWalker(RandomWalker):

    def __init__(self, q, field):
        super(Edge2VecWalker, self).__init__()
        self.q = q
        self.field = field

    def generate_walks(self, graph, walk_length, num_walks):
        clusters = self._create_clusters(graph)

        neighborhoods = [self._create_neighborhoods(graph, prev_level=None)]
        edge_features = self._get_edge_features(graph)
        distances = [self._compute_distances(graph, clusters, neighborhoods[0], edge_features, None)]

        walks = []
        for _ in range(num_walks):
            for edge in graph.edges():
                walk = [edge]
                k = 0
                e = edge
                while len(walk) < walk_length:
                    should_stay = np.random.random() < self.q
                    if not should_stay:
                        should_move_up = np.random.random() < 0.5
                        if should_move_up or k == 0:
                            k += 1
                        else:
                            k -= 1

                        if len(neighborhoods) >= k:
                            kth_neighborhood = self._create_neighborhoods(graph, neighborhoods[k-1])
                            neighborhoods.append(kth_neighborhood)

                            kth_distances = self._compute_distances(graph=graph,
                                                                    clusters=clusters,
                                                                    neighborhoods=kth_neighborhood,
                                                                    features=edge_features,
                                                                    prev_layer_distances=distances[k-1])
                            distances.append(kth_distances)
                    else:
                        weights = neg_softmax(distances[k][e])
                        next_edge_index = np.random.choice(len(clusters[e]), p=weights)
                        e = clusters[e][next_edge_index]
                        walk.append(e)
                walks.append(walk)
        return walks

    def _create_clusters(self, graph):
        clusters = {}
        edge_list = list(graph.edges())
        for e in graph.edges():
            clusters[e] = edge_list
        return clusters

    def _create_neighborhoods(self, graph, prev_level):
        neighborhoods = {}
        for e in graph.edges():
            prev = prev_level[e] if prev_level != None else [e]
            neighborhood = set()
            for src, dest in prev:
                for n in graph.neighbors(src):
                    neighborhood.add((src, n))
                for n in graph.neighbors(dest):
                    neighborhood.add((dest, n))
            neighborhoods[e] = neighborhood
        return neighborhoods

    def _get_edge_features(self, graph):
        features = {}
        for src, dest in graph.edges():
            # Excludes the current edge
            src_deg = graph.degree(src) - 1
            dest_deg = graph.degree(dest) - 1

            weight = graph[src][dest][self.field] if self.field in graph[src][dest] else 1
            features[(src, dest)] = np.array([src_deg+dest_deg, weight])
        return features

    def _get_features_for_neighborhood(self, neighborhood, features):
        neigh_features = [features[edge] for edge in neighborhood]
        return list(sorted(neigh_features, key=lambda t: t[0]))

    def _compute_distances(self, graph, clusters, neighborhoods, features, prev_layer_distances):
        distances = {}
        for edge in graph.edges():
            edge_dist = {}
            edge_features = self._get_features_for_neighborhood(neighborhoods[edge], features)
            for e in clusters[edge]:
                e_features = self._get_features_for_neighborhood(neighborhoods[e], features)
                prev_layer_dist = prev_layer_distances[edge][e] if prev_layer_distances != None else 0
                edge_dist[e] = self._compute_distance(edge_features, e_features, prev_layer_dist)
            distances[edge] = edge_dist
        return distances

    def _compute_distance(self, nf1, nf2, prev_layer_distance):
        dtw_dist = fastdtw(nf1, nf2, dist=lambda x,y: np.linalg.norm(x - y))
        return prev_layer_distance + dtw_dist[0]

