import tensorflow as tf
import numpy as np
import networkx as nx
from datetime import datetime
from os import mkdir
from embedding_model import EmbeddingModel
from random_walkers import RandomWalkerFactory
from utils import create_mini_batches, append_to_log, create_walk_windows
from utils import cluster_nodes, cluster_edges, create_negative_samples
from utils import append_lines_to_file, load_data_file, lst_elems_to_str
from plot import plot_graph
from constants import *


class GraphEmbedder:

    def __init__(self, graph, params):
        self.graph = graph
        self.params = params

        # Create and initialize embedding model
        self.model = EmbeddingModel(params=params)
        self.nodes_ph = self.model.create_placeholder(dtype=tf.int32, shape=[None], name='nodes-ph')
        self.walks_ph = self.model.create_placeholder(dtype=tf.int32, shape=[None, params['window_size']],
                                                      name='walks-ph')
        self.neg_samples_ph = self.model.create_placeholder(dtype=tf.int32, shape=[None, params['neg_samples']],
                                                            name='neg-sample-ph')
        self.model.build(nodes=self.nodes_ph,
                         walks=self.walks_ph,
                         neg_samples=self.neg_samples_ph,
                         num_nodes=graph.number_of_nodes())
        self.model.init()

        # Make output folder for this model
        timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        self.output_folder = params['output_folder'] + params['graph_name'] + '-' + timestamp + '/'
        mkdir(self.output_folder)

    def generate_walks(self):
        # Create the random walker for this instance
        random_walker = RandomWalkerFactory.get_random_walker(name=self.params['walker_type'],
                                                              options=self.params['walker_options'])
        if random_walker == None:
            raise ValueError('Could not find the random walker named {0}.'.format(params['walker_type']))

        walks_file = self.params['data_folder'] + '/walks.txt'
        nodes_file = self.params['data_folder'] + '/nodes.txt'

        # Execute random walks
        walks = random_walker.generate_walks(graph=self.graph,
                                             walk_length=self.params['walk_length'],
                                             num_walks=self.params['num_walks'])

        data_windows = create_walk_windows(walks=walks, window_size=self.params['window_size'])
        for i in range(0, len(data_windows[WALKS]), WRITE_AMOUNT):
            walk_window = [' '.join(lst_elems_to_str(w)) for w in data_windows[WALKS][i:i+WRITE_AMOUNT]]
            node_window = [str(n) for n in data_windows[NODES][i:i+WRITE_AMOUNT]]

            append_lines_to_file(walk_window, walks_file)
            append_lines_to_file(node_window, nodes_file)

    def train(self):

        # Load input dataset
        data_windows = {
            WALKS: load_data_file(self.params['data_folder'] + '/walks.txt'),
            NODES: load_data_file(self.params['data_folder'] + '/nodes.txt')
        }

        # Append heading to the log file
        log_path = self.output_folder + 'log.csv'
        append_to_log(['Epoch', 'Average Loss per Sample'], log_path)
        
        # Get a list of the graph nodes for negative sampling
        graph_nodes = list(self.graph.nodes())

        convergence_count = 0
        prev_loss = BIG_NUMBER
        for epoch in range(self.params['epochs']):

            # Create data batches
            neg_samples = create_negative_samples(nodes=graph_nodes,
                                                  walk_windows=data_windows[WALKS],
                                                  node_windows=data_windows[NODES],
                                                  num_neg_samples=self.params['neg_samples'])
            batches = create_mini_batches(walks=data_windows[WALKS],
                                          nodes=data_windows[NODES],
                                          neg_samples=neg_samples,
                                          batch_size=self.params['batch_size'])
            walk_batches = batches[WALKS]
            node_batches = batches[NODES]
            neg_sample_batches = batches[NEG_SAMPLES]

            losses = []
            for walk_batch, node_batch, neg_sample_batch in zip(walk_batches, node_batches, neg_sample_batches):
                feed_dict = {
                    self.nodes_ph: node_batch,
                    self.walks_ph: walk_batch,
                    self.neg_samples_ph: neg_sample_batch
                }
                loss = self.model.run_train_step(feed_dict)
                losses.append(loss / float(len(walk_batch)))

            # Compute average losses and output to log file
            avg_loss = np.average(losses)
            print('Average loss for epoch {0}: {1}'.format(epoch, avg_loss))

            append_to_log([epoch, avg_loss], log_path)

            # Early stopping
            if abs(prev_loss - avg_loss) < SMALL_NUMBER or prev_loss < avg_loss:
                convergence_count += 1
            else:
                convergence_count = 0

            if convergence_count == self.params['patience']:
                print('Early Stopping.')
                break

            prev_loss = avg_loss

        # Save Model
        self.model.save(self.output_folder)

    def compute_clusters(self):
        # Compute embeddings
        feed_dict = {
            self.nodes_ph: np.array(self.graph.nodes())
        }
        node_embeddings, edge_embeddings = self.model.inference(feed_dict)
        clustered_graph = cluster_edges(self.graph, edge_embeddings, self.params['num_clusters'])
        clustered_graph = cluster_nodes(self.graph, node_embeddings, self.params['num_clusters'])

        # Visualization using GraphViz is best for smaller graphs (<50 nodes)
        if self.params['visualize']:
            output_file = self.output_folder + self.params['graph_name'] + '.png'
            plot_graph(clustered_graph, self.params['num_clusters'], output_file)

        # Write output graph to Graph XML
        output_file = self.output_folder + self.params['graph_name'] + '.gexf'
        nx.write_gexf(clustered_graph, output_file)

