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

                # Tensor Dims: B x R x L x D
                node_shape = tf.shape(node_embeddings)
                node_expanded = tf.reshape(node_embeddings, [node_shape[0], 1, 1, node_shape[1]])
                walk_multiply = node_expanded * walk_embeddings

                # Tensor Dims: B x 1
                neigh_sum = tf.reduce_sum(walk_multiply, axis=[2, 3])
                neigh_mean = tf.reduce_mean(neigh_sum, axis=1)

                # Tensor Dims: B x 1 x D
                node_expanded = tf.reshape(node_embeddings, [node_shape[0], 1, node_shape[1]])

                # Tensor Dims: B x B x D
                node_multiply = node_expanded * tf.transpose(node_expanded, perm=[1, 0, 2])

                # Tensor Dims: B x 1
                node_dot_prod = tf.exp(tf.reduce_sum(node_multiply, axis=2))
                node_neighborhood = tf.reduce_sum(node_dot_prod, axis=1)

                self.loss_op = -tf.reduce_sum(-tf.log(node_neighborhood) + neigh_mean)
                self.output_ops.append(node_embeddings)
