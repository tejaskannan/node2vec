import networkx as nx
import numpy as np
from constants import *
from fastdtw import fastdtw
from utils import dist, compressed_degree_list


class RandomWalkerFactory:

    def get_random_walker(name, options):
        if name == 'node2vec':
            assert 'p' in options and 'q' in options, 'Must specify both p and q.'
            return Node2VecWalker(p=options['p'], q=options['q'])
        if name == 'struc2vec':
            assert 'q' in options
            k_max = options['k_max'] if 'k_max' in options else -1
            return Struc2VecWalker(q=options['q'], k_max=k_max)
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
                    neighbor_weights = np.array(neighbor_weights) / (np.sum(neighbor_weights) + SMALL_NUMBER)

                    # Move the previous pointer to the current node after the first iteration
                    if len(walk) > 0:
                        t = v

                    # Select the next node
                    if abs(1.0 - np.sum(neighbor_weights)) < 1e-3:
                        v = np.random.choice(list(graph.neighbors(v)), p=neighbor_weights)

                walks.append(walk)
        return np.array(walks)


class Struc2VecWalker(RandomWalker):

    def __init__(self, q, k_max):
        super(Struc2VecWalker, self).__init__()
        self.q = q
        self.k_max = k_max

    def generate_walks(self, graph, walk_length, num_walks):

        # Vertices which are direct neighbros of each node
        neighborhoods = {
            node: {0: set([node])} for node in graph.nodes()
        }

        # Create degree lists for the 0th level
        degree_neighborhoods = {node: {} for node in graph.nodes()}
        self._add_kth_degree_neighborhood(graph, degree_neighborhoods, neighborhoods, 0)

        # Initialize 0th level weights
        l0_weight, l0_avg = self._compute_weights(graph, degree_neighborhoods, 0, None)
        weights = [l0_weight]
        avg_weights = [l0_avg]

        nodes_lst = list(graph.nodes())
        walks = []
        for _ in range(num_walks):
            for node in graph.nodes():
                walk = [node]
                k = 0
                u = node
                while len(walk) < walk_length:
                    should_stay = np.random.random() < self.q
                    if should_stay:
                        u = np.random.choice(nodes_lst, p=weights[k][u])
                        walk.append(u)
                    else:
                        if k == 0:
                            k += 1
                        elif self.k_max != -1 and k == self.k_max:
                            k -= 1
                        else:
                            gamma = np.sum([int(weights[k][u][v] > avg_weights[k]) for v in graph.nodes()])
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
                            lk_weights, avg = self._compute_weights(graph, degree_neighborhoods, k, weights[k-1])
                            weights.append(lk_weights)
                            avg_weights.append(avg)

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

    # O(n^2) algorithm to compute the weights for level k
    def _compute_weights(self, graph, degree_neighborhoods, k, prev_layer_weights):
        weights = []
        total_weight = 0.0
        for u in graph.nodes():
            w = [self._compute_weight(u, v, degree_neighborhoods, k, prev_layer_weights) \
                 for v in graph.nodes()]
            w = np.exp(w)

            # Prevent moving back to the same node
            w[u] = 0.0
            
            # Normalize the weights
            t = np.sum(w)
            if t < SMALL_NUMBER:
                w[u] = 1.0
            else:
                w = w / t

            total_weight += t
            weights.append(w)

        return np.array(weights), np.average(weights)

    def _compute_weight(self, u, v, degree_neighborhoods, k, prev_layer_weights):
        if u == v:
            return 0.0
        distance, _ = fastdtw(degree_neighborhoods[u][k], degree_neighborhoods[v][k], dist=dist)
        f_k = prev_layer_weights[u, v] + distance if prev_layer_weights is not None else distance
        return -f_k
