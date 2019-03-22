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

        # Walks is a placeholder holding the set of walks (B x L)
        walks = kwargs['walks']

        # Number of nodes in the graph
        num_nodes = kwargs['num_nodes']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                node_init = tf.random.uniform(shape=(num_nodes, self.params['embedding_size']), maxval=1.0)
                node_embedding_var = tf.Variable(node_init, name='node-embedding-var')

                # Tensor Dims: B x D where D is the embedding size
                node_embeddings = tf.nn.embedding_lookup(node_embedding_var, nodes, name='node-embeddings')

                # Tensor Dims: B x L x D
                walk_embeddings = tf.nn.embedding_lookup(node_embedding_var, walks, name='walk-embeddings')

                # Tensor Dims: B x L x D
                node_expanded = tf.expand_dims(node_embeddings, axis=2)
                walk_multiply = node_expanded * tf.transpose(walk_embeddings, [0, 2, 1])

                # Tensor Dims: B x 1
                neigh_sum = tf.reduce_sum(walk_multiply, axis=[1, 2])

                self.loss_op = tf.reduce_sum(neigh_sum)
                self.output_ops.append(node_embeddings)
