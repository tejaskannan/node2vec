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
from utils import lst_pairs_to_str, edge_index
from plot import plot_graph
from constants import *
from annoy import AnnoyIndex


class GraphEmbedder:

    def __init__(self, graph, params):
        self.graph = graph
        self.params = params

        # Create and initialize embedding model
        self.model = EmbeddingModel(params=params)
        self.points_ph = self.model.create_placeholder(dtype=tf.int32, shape=[None], name='points-ph')
        self.walks_ph = self.model.create_placeholder(dtype=tf.int32, shape=[None, params['window_size']],
                                                      name='walks-ph')
        self.neg_samples_ph = self.model.create_placeholder(dtype=tf.int32, shape=[None, params['neg_samples']],
                                                            name='neg-sample-ph')

        self.use_edges = params['walker_type'] == 'edge2vec'
        num_points = graph.number_of_edges() if self.use_edges else graph.number_of_nodes()
        self.model.build(points=self.points_ph,
                         walks=self.walks_ph,
                         neg_samples=self.neg_samples_ph,
                         num_points=num_points)
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
        points_file = self.params['data_folder'] + '/points.txt'

        # Execute random walks
        walks = random_walker.generate_walks(graph=self.graph,
                                             walk_length=self.params['walk_length'],
                                             num_walks=self.params['num_walks'])

        data_windows = create_walk_windows(walks=walks, window_size=self.params['window_size'])
        for i in range(0, len(data_windows[WALKS]), WRITE_AMOUNT):
            if self.params['walker_type'] == 'edge2vec':
                walk_window = [' '.join(lst_pairs_to_str(w)) for w in data_windows[WALKS][i:i+WRITE_AMOUNT]]
                points_window = [str(e[0]) + '-' + str(e[1]) for e in data_windows[POINTS][i:i+WRITE_AMOUNT]]
            else:
                walk_window = [' '.join(lst_elems_to_str(w)) for w in data_windows[WALKS][i:i+WRITE_AMOUNT]]
                points_window = [str(n) for n in data_windows[POINTS][i:i+WRITE_AMOUNT]]

            append_lines_to_file(walk_window, walks_file)
            append_lines_to_file(points_window, points_file)

    def train(self):

        # Load input dataset
        data_windows = {
            WALKS: load_data_file(self.params['data_folder'] + '/walks.txt', self.graph, self.use_edges),
            POINTS: load_data_file(self.params['data_folder'] + '/points.txt', self.graph, self.use_edges)
        }

        # Append heading to the log file
        log_path = self.output_folder + 'log.csv'
        append_to_log(['Epoch', 'Average Loss per Sample'], log_path)
        
        # Get a list of the graph points for negative sampling
        graph_points = list(self.graph.edges()) if self.use_edges else list(self.graph.nodes())

        convergence_count = 0
        prev_loss = BIG_NUMBER
        for epoch in range(self.params['epochs']):

            # Create data batches
            neg_samples = create_negative_samples(points=graph_points,
                                                  walk_windows=data_windows[WALKS],
                                                  point_windows=data_windows[POINTS],
                                                  num_neg_samples=self.params['neg_samples'],
                                                  graph=self.graph,
                                                  use_edges=self.use_edges)
            batches = create_mini_batches(walks=data_windows[WALKS],
                                          points=data_windows[POINTS],
                                          neg_samples=neg_samples,
                                          batch_size=self.params['batch_size'])
            walk_batches = batches[WALKS]
            point_batches = batches[POINTS]
            neg_sample_batches = batches[NEG_SAMPLES]

            losses = []
            for walk_batch, point_batch, neg_sample_batch in zip(walk_batches, point_batches, neg_sample_batches):
                feed_dict = {
                    self.points_ph: point_batch,
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
        if self.use_edges:
            index = edge_index(self.graph)
            points = np.array([index[e] for e in self.graph.edges()])
        else:
            points = np.array(self.graph.nodes())

        feed_dict = {
            self.points_ph: points
        }

        point_embeddings = self.model.inference(feed_dict)[0]
        if self.use_edges:
            clustered_graph = cluster_edges(self.graph, point_embeddings, self.params['num_clusters'])
        else:
            clustered_graph = cluster_nodes(self.graph, point_embeddings, self.params['num_clusters'])

        print(point_embeddings)

        # Visualization using GraphViz is best for smaller graphs (<50 nodes)
        if self.params['visualize']:
            output_file = self.output_folder + self.params['graph_name'] + '.png'
            plot_graph(clustered_graph, self.params['num_clusters'], output_file, self.use_edges)

        # Write output graph to Graph XML
        output_file = self.output_folder + self.params['graph_name'] + '.gexf'
        nx.write_gexf(clustered_graph, output_file)

        # Write embeddings to an Annoy index
        annoy_index = AnnoyIndex(self.params['embedding_size'])
        for i, embedding in enumerate(point_embeddings):
            annoy_index.add_item(i, embedding)
        annoy_index.build(1)
        annoy_index.save(self.output_folder + self.params['graph_name'] + '.ann')
