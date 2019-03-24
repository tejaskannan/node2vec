import argparse
import networkx as nx
from utils import load_params
from load import load_to_networkx
from os.path import exists
from graph_embedder import GraphEmbedder


def load_graph(graph_name, graph_options):
    if graph_name == 'barbell':
        assert 'm1' in graph_options and 'm2' in graph_options
        return nx.barbell_graph(graph_options['m1'], graph_options['m2'])
    elif graph_name == 'lollipop':
        assert 'm' in graph_options and 'n' in graph_options
        return nx.barbell_graph(graph_options['m'], graph_options['n'])
    elif graph_name == 'star':
        assert 'n' in graph_options
        return nx.barbell_graph(graph_options['n'])

    # Any other graph name is assumed to be a road graph
    graph_path = 'graphs/{0}.tntp'.format(graph_name)
    if not exists(graph_path):
        print('The graph {0} does not exist.'.format(graph_path))
        return
    return load_to_networkx(graph_path)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Computing Min Cost Flows using Graph Neural Networks.')
    parser.add_argument('--params', type=str, help='Parameters JSON file.')
    parser.add_argument('--train', action='store_true', help='Flag for training the embedding model.')
    parser.add_argument('--generate', action='store_true', help='Flag for generating a new dataset.')

    args = parser.parse_args()

    if not exists(args.params):
        print('The file {0} does not exist.'.format(args.params))
        return

    # Fetch parameters
    params = load_params(args.params)
    
    # Load graph
    options = params['graph_options'] if 'graph_options' in params else None
    graph = load_graph(params['graph_name'], options)

    # Train and test the embedding model
    embedder = GraphEmbedder(graph, params)

    if args.train:
        embedder.train()
        embedder.compute_clusters()
    elif args.generate:
        embedder.generate_walks()


if __name__ == '__main__':
    main()
