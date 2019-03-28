import tensorflow as tf
import numpy as np
from base_model import Model
from constants import *


class EmbeddingModel(Model):

    def __init__(self, params, name='node2vec'):
        super(EmbeddingModel, self).__init__(params, name)

    def build(self, **kwargs):
        
        # Tensor of integers representing point indices (B, )
        points = kwargs['points']

        # Walks is a placeholder holding the set of walks (B x L)
        walks = kwargs['walks']

        # Tensor denoting points used for negative sampling (B x S)
        neg_samples = kwargs['neg_samples']

        # Number of points in the graph
        num_points = kwargs['num_points']

        with self._sess.graph.as_default():
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

                var_init = tf.random.normal(shape=(num_points, self.params['embedding_size']))
                point_embedding_var = tf.Variable(var_init, name='point-embedding-var')

                # Tensor Dims: B x D where D is the embedding size
                point_embeddings = tf.nn.embedding_lookup(point_embedding_var, points, name='embeddings',
                                                          max_norm=1.0)

                # Tensor Dims: B x L x D
                walk_embeddings = tf.nn.embedding_lookup(point_embedding_var, walks, name='embeddings',
                                                         max_norm=1.0)

                # Tensor Dims: B x S x D
                neg_sample_embeddings = tf.nn.embedding_lookup(point_embedding_var, neg_samples,
                                                               name='embeddings',
                                                               max_norm=1.0)

                # Tensor Dims: B x L x D
                point_shape = tf.shape(point_embeddings)
                point_expanded = tf.reshape(point_embeddings, [point_shape[0], 1, point_shape[1]])
                walk_multiply = point_expanded * walk_embeddings

                # Tensor Dims: B x 1
                neighborhood_sum = tf.reduce_sum(walk_multiply, axis=[1, 2])

                # Tensor Dims: B x S
                neg_sample_similarity = tf.reduce_sum(point_expanded * neg_sample_embeddings, axis=2)
                neg_sample_sum = tf.reduce_sum(tf.exp(neg_sample_similarity), axis=1)

                self.loss_op = tf.reduce_sum(tf.log(neg_sample_sum) - neighborhood_sum)
                self.output_ops.append(point_embeddings)
