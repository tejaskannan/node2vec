import tensorflow as tf
import numpy as np
from os.path import exists
from os import mkdir
from constants import *


class Model:

    def __init__(self, params, name):
        self.name = name
        self.params = params
        self._sess = tf.Session(graph=tf.Graph())
        self.optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])

        # Must be set by a concrete subclass
        self.loss_op = None
        self.output_ops = []

    def init(self):
        with self._sess.graph.as_default():
            self._build_optimizer_op()
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)

    def build(self, **kwargs):
        raise NotImplementedError()

    def run_train_step(self, feed_dict):
        with self._sess.graph.as_default():
            ops = [self.loss_op, self.optimizer_op]
            op_result = self._sess.run(ops, feed_dict=feed_dict)
            return op_result[0]

    def inference(self, feed_dict):
        with self._sess.graph.as_default():
            op_results = self._sess.run(self.output_ops, feed_dict=feed_dict)
            return op_results

    def _build_optimizer_op(self):
        trainable_vars = self._sess.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(self.loss_op, trainable_vars)
        clipped_grad, _ = tf.clip_by_global_norm(gradients, self.params['gradient_clip'])
        pruned_gradients = []
        for grad, var in zip(clipped_grad, trainable_vars):
            if grad is not None:
                pruned_gradients.append((grad, var))

        self.optimizer_op = self.optimizer.apply_gradients(pruned_gradients)

    def save(self):
        out_folder = self.params['output_folder']
        if not exists(out_folder):
            mkdir(save_folder)

        params_path = out_folder + '/params.pkl.gz'
        with gzip.GzipFile(params_path, 'wb') as out_file:
            pickle.dump(self.params, out_file)

        model_path = out_folder + '/model-' + self.name + '.ckpt'
        with self._sess.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self._sess, model_path)

    def restore(self, save_folder):
        params_path = save_folder + '/params.pkl.gz'
        with gzip.GzipFile(params_path, 'rb') as params_file:
            params_dict = pickle.load(params_file)

        self.params = params_dict

        model_path = save_folder + '/model-' + self.name + '.ckpt'
        with self._sess.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self._sess, model_path)
