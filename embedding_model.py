import tensorflow as tf
import numpy as np
from base_model import Model
from constants import *


class EmbeddingModel(Model):

    def __init__(self, params, name='node2vec'):
        super(EmbeddingModel, self).__init__(params, name)

    def build(self, **kwargs):
        
        # Tensor of integers representing node indices (B, )
        nodes = kwargs['nodes']

        # Walks is a placeholder holding the set of walks (B x R x L)
        walks = kwargs['walks']

        # Number of nodes in the graph
        num_nodes = kwargs['num_nodes']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                node_init = tf.random.uniform(shape=(num_nodes, self.params['embedding_size']), maxval=1.0)
                node_embedding_var = tf.Variable(node_init, name='node-embedding-var')

                # Tensor Dims: B x D where D is the embedding size
                node_embeddings = tf.nn.embedding_lookup(node_embedding_var, nodes, name='node-embeddings')

                # Tensor Dims: B x R x L x D
                walk_embeddings = tf.nn.embedding_lookup(node_embedding_var, walks, name='walk-embeddings')
                print(walk_embeddings)

                # Tensor Dims: B x R x L x D
                node_shape = tf.shape(node_embeddings)
                node_expanded = tf.reshape(node_embeddings, [node_shape[0], 1, 1, node_shape[1]])
                walk_multiply = node_expanded * walk_embeddings

                # Tensor Dims: B x 1
                neigh_sum = tf.reduce_sum(walk_multiply, axis=[1, 2, 3])

                self.loss_op = tf.reduce_sum(neigh_sum)
                self.output_ops.append(node_embeddings)
