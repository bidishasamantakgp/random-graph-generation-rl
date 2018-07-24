from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class VAEGRLCell(object):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, weight, features, z_dim, bin_dim, enc_mu, enc_sigma, edges, index):
        '''
        Args:
        adj : adjacency matrix
	    features: feature matrix
	    '''
        self.adj = adj
        self.features = features
        self.z_dim = z_dim
        self.weight = weight
        self.name = self.__class__.__name__.lower()
        self.bin_dim = bin_dim
        self.enc_mu = enc_mu
        self.enc_sigma = enc_sigma
        self.edges = edges
        self.index = index

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, c_x, n, d, k, eps_passed, sample, scope=None):
        '''
		Args:
			c_x - tensor to be filled up with random walk property
			n - number of nodes
			d - number of features in the feature matrix
			k - length of random walk
				defaults to be None
    	'''
        with tf.variable_scope(scope or type(self).__name__):

            # Random sampling ~ N(0, 1)
            eps = eps_passed

            temp_stack = []
            for i in range(n):
                temp_stack.append(tf.matmul(self.enc_sigma[i], eps[i]))

            z = tf.add(self.enc_mu, tf.stack(temp_stack))
            # While we are trying to sample some edges, we sample Z from prior
            if sample:
                z = eps
            #############negative sampling################

            # index : negative index
            # edge_list : all edges
            edges = tf.gather(self.edges, self.index)
            ##This is the set of edges which has
            list_edges = tf.unstack(edges)

            with tf.variable_scope("DecoderRL"):
                z_stack = []
                z_stack_weight = []
                z_stack_label = []

                neg_edges = tf.random_shuffle(self.neg_edges)

                # sampled edges are here
                for edge in list_edges.extend(neg_edges[:10]):
                    z_stack.append(tf.concat(values=(tf.transpose(edge[0]), tf.transpose(edge[1])), axis=1)[0])
                    for j in range(self.bin_dim):
                                m = np.zeros((1, self.bin_dim))
                                m[0][j] = 1
                                z_stack_weight.append(tf.concat(values=(tf.transpose(edge[0]), tf.transpose(edge[1]), m), axis=1)[0])

                dec_hidden = fc_layer(tf.stack(z_stack), 1, activation=tf.nn.softplus, scope="hidden")
                weight = fc_layer(tf.stack(z_stack_weight), 1, activation=tf.nn.softplus, scope="marker")
                label = fc_layer(tf.stack(z_stack_label), 1, activation=tf.nn.softplus, scope="label")

        return (dec_hidden, weight)

    def call(self, inputs, n, d, k, eps_passed, sample):
        return self.__call__(inputs, n, d, k, eps_passed, sample)
