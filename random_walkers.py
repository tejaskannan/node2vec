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
            assert 'q' in options and 'k_max' in options, 'Must specify both q and k_max'
            return Struc2VecWalker(q=options['q'], k_max=options['k_max'])
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
        assert k_max > 1, 'k_max must be greater than 1'
        self.q = q
        self.k_max = k_max

    def generate_walks(self, graph, walk_length, num_walks):

        degree_neighborhoods = self._create_kth_degree_neighborhoods(graph)
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
                        v = np.random.choice(nodes_lst, p=weights[k][u])
                        walk.append(v)
                    else:
                        if k == 0:
                            k += 1
                        elif k == self.k_max:
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

                        # Only compute a layers weights when the layer is reached
                        if len(weights) <= k:
                            lk_weights, avg = self._compute_weights(graph, degree_neighborhoods, k, weights[k-1])
                            weights.append(lk_weights)
                            avg_weights.append(avg)
                walks.append(walk)
        return np.array(walks)

    # O(nk) procedure to fetch the compressed degree lists for each node at each level
    def _create_kth_degree_neighborhoods(self, graph):
        # Construct kth-order neighborhoods
        neighborhoods = {
            node: {0: list(graph.neighbors(node))} for node in graph.nodes()
        }
        for k in range(1, self.k_max + 1):
            for node in graph.nodes():
                prev = neighborhoods[node][k-1]
                neighbors = []
                for n in prev:
                    neighbors += list(graph.neighbors(n))
                neighborhoods[node][k] = neighbors

        degree_neighborhoods = {node: {} for node in graph.nodes()}
        for k in range(0, self.k_max + 1):
            for node in graph.nodes():
                degree_neighborhoods[node][k] = compressed_degree_list(graph, neighborhoods[node][k])
        print('Created Degree neighborhoods')
        return degree_neighborhoods

    # O(n^2) algorithm to compute the weights for level k
    def _compute_weights(self, graph, degree_neighborhoods, k, prev_layer_weights):
        weights = []
        total_weight = 0.0
        for u in graph.nodes():
            w = [self._compute_weight(u, v, degree_neighborhoods, k, prev_layer_weights) \
                 for v in graph.nodes()]
            w = np.exp(w)
            
            # Normalize the weights
            total_weight = np.sum(w)
            w = w / (total_weight + SMALL_NUMBER)

            if abs(np.sum(w) - 1.0) > 1e-3:
                print(np.sum(w))

            weights.append(w)

        num_edges = pow(graph.number_of_nodes(), 2)
        return np.array(weights), (total_weight / num_edges)

    def _compute_weight(self, u, v, degree_neighborhoods, k, prev_layer_weights):
        if u == v:
            return 0.0
        distance, _ = fastdtw(degree_neighborhoods[u][k], degree_neighborhoods[v][k], dist=dist)
        f_k = prev_layer_weights[u, v] + distance if prev_layer_weights is not None else 0.0
        return -f_k
