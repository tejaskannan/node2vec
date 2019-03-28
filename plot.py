import networkx as nx
from matplotlib import cm
from matplotlib import colors
import numpy as np


def plot_graph(graph, num_clusters, file_path, use_edges):
    cmap = cm.get_cmap(name='viridis')
    color_series = np.linspace(start=0.0, stop=1.0, num=num_clusters)

    agraph = nx.drawing.nx_agraph.to_agraph(graph)

    if use_edges:
        for src, dest, cluster in graph.edges.data('cluster'):
            e = agraph.get_edge(src, dest)
            e.attr['color'] = colors.rgb2hex(cmap(color_series[cluster])[:3])
    else:
        for node, cluster in graph.nodes.data('cluster'):
            n = agraph.get_node(node)
            n.attr['color'] = colors.rgb2hex(cmap(color_series[cluster])[:3])

    agraph.draw(file_path, prog='dot')
