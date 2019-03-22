import argparse
from utils import load_params
from load import load_to_networkx
from os.path import exists
from graph_embedder import GraphEmbedder


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

    # Train and test the embedding model
    embedder = GraphEmbedder(graph, params)
    embedder.train()
    embedder.compute_clusters()


if __name__ == '__main__':
    main()
