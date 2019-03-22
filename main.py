import argparse
import networkx as nx
import tensorflow as tf
import numpy as np
from embedding_model import EmbeddingModel
from utils import load_params, create_nodes_tensor, create_random_walks
from utils import create_mini_batches
from load import load_to_networkx
from os.path import exists
from constants import *


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Computing Min Cost Flows using Graph Neural Networks.')
    parser.add_argument('--params', type=str, help='Parameters JSON file.')
    args = parser.parse_args()

    if not exists(args.params):
        print('The file {0} does not exist.'.format(args.params))
        return

    # Fetch parameters
    params = load_params(args.params)

    graph_path = 'graphs/{0}.tntp'.format(params['graph_name'])
    if not exists(graph_path):
        print('The graph {0} does not exist.'.format(graph_path))
        return
    
    # Load graph
    graph = load_to_networkx(graph_path)

    model = EmbeddingModel(params=params)

    # Create placeholders
    with model._sess.graph.as_default():
        nodes_ph = tf.placeholder(tf.int32, shape=[None], name='nodes-ph')
        walks_ph = tf.placeholder(tf.int32, shape=[None, params['walk_length']], name='walks-ph')

    model.build(nodes=nodes_ph, walks=walks_ph, num_nodes=graph.number_of_nodes())
    model.init()

    walks = create_random_walks(graph=graph,
                                length=params['walk_length'],
                                p=params['p'],
                                q=params['q'])
    nodes = create_nodes_tensor(graph=graph)

    convergence_count = 0
    prev_loss = BIG_NUMBER
    for epoch in range(params['epochs']):

        # Create data batches
        batches = create_mini_batches(walks=walks, nodes=nodes, batch_size=params['batch_size'])
        walk_batches = batches[WALKS]
        node_batches = batches[NODES]

        losses = []
        for walk_batch, node_batch in zip(walk_batches, node_batches):
            feed_dict = {
                nodes_ph: node_batch,
                walks_ph: walk_batch
            }
            loss = model.run_train_step(feed_dict)
            losses.append(loss)

        avg_loss = np.average(losses)
        print('Average loss for epoch {0}: {1}'.format(epoch, avg_loss))

        if abs(prev_loss - avg_loss) < SMALL_NUMBER:
            convergence_count += 1

        prev_loss = avg_loss
        if convergence_count == params['patience']:
            print('Early Stopping.')
            break




if __name__ == '__main__':
    main()
