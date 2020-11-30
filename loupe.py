import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

class PoolingBaseModel(object):
    """Inherit from this class when implementing new models."""

    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
            gating=True,add_batch_norm=True, is_training=True, suffix_tensor_name=""):
        """Initialize a NetVLAD block.

        Args:
        feature_size: Dimensionality of the input features.
        max_samples: The maximum number of samples to pool.
        cluster_size: The number of clusters.
        output_dim: size of the output space after dimension reduction.
        add_batch_norm: (bool) if True, adds batch normalization.
        is_training: (bool) Whether or not the graph is training.
        """

        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.suffix_tensor_name = suffix_tensor_name

    def forward(self, reshaped_input):
        raise NotImplementedError("Models should implement the forward pass.")

    def context_gating(self, input_layer):
        """Context Gating

        Args:
        input_layer: Input layer in the following shape:
        'batch_size' x 'number_of_activation'

        Returns:
        activation: gated layer in the following shape:
        'batch_size' x 'number_of_activation'
        """

        input_dim = input_layer.get_shape().as_list()[1]

        gating_weights = tf.get_variable("gating_weights" + self.suffix_tensor_name,
          [input_dim, input_dim],
          initializer = tf.random_normal_initializer(
          stddev=1 / math.sqrt(input_dim)))

        gates = tf.matmul(input_layer, gating_weights)

        if self.add_batch_norm:
          gates = slim.batch_norm(
              gates,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="gating_bn" + self.suffix_tensor_name)
        else:
          gating_biases = tf.get_variable("gating_biases" + self.suffix_tensor_name,
            [input_dim],
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
          gates += gating_biases

        gates = tf.sigmoid(gates)

        activation = tf.multiply(input_layer,gates)

        return activation

class NetRVLAD(PoolingBaseModel):
    """Creates a NetRVLAD class (Residual-less NetVLAD).
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
            gating=True,add_batch_norm=True, is_training=True, suffix_tensor_name=""):
        super(self.__class__, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
            suffix_tensor_name=suffix_tensor_name)

    def forward(self, reshaped_input):
        """Forward pass of a NetRVLAD block.

        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])

        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """


        cluster_weights = tf.get_variable("cluster_weights" + self.suffix_tensor_name,
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(
                  stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn" + self.suffix_tensor_name)
        else:
          cluster_biases = tf.get_variable("cluster_biases" + self.suffix_tensor_name,
            [self.cluster_size],
            initializer = tf.random_normal_initializer(
                stddev=1 / math.sqrt(self.feature_size)))
          tf.summary.histogram("cluster_biases" + self.suffix_tensor_name, cluster_biases)
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        activation = tf.transpose(activation,perm=[0,2,1])

        reshaped_input = tf.reshape(reshaped_input,[-1,
            self.max_samples, self.feature_size])
        vlad = tf.matmul(activation,reshaped_input)

        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1,self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        hidden1_weights = tf.get_variable("hidden1_weights" + self.suffix_tensor_name,
          [self.cluster_size*self.feature_size, self.output_dim],
          initializer=tf.random_normal_initializer(
              stddev=1 / math.sqrt(self.cluster_size)))

        vlad = tf.matmul(vlad, hidden1_weights)

        if self.gating:
            vlad = super(self.__class__, self).context_gating(vlad)


        return vlad

class NetVLAD(PoolingBaseModel):
    """Creates a NetVLAD class.
    """
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
            gating=True,add_batch_norm=True, is_training=True, suffix_tensor_name=""):
        super(self.__class__, self).__init__(
            feature_size=feature_size,
            max_samples=max_samples,
            cluster_size=cluster_size,
            output_dim=output_dim,
            gating=gating,
            add_batch_norm=add_batch_norm,
            is_training=is_training,
            suffix_tensor_name=suffix_tensor_name)

    def forward(self, reshaped_input):
        """Forward pass of a NetVLAD block.
        Args:
        reshaped_input: If your input is in that form:
        'batch_size' x 'max_samples' x 'feature_size'
        It should be reshaped in the following form:
        'batch_size*max_samples' x 'feature_size'
        by performing:
        reshaped_input = tf.reshape(input, [-1, features_size])
        Returns:
        vlad: the pooled vector of size: 'batch_size' x 'output_dim'
        """


        cluster_weights = tf.get_variable("cluster_weights" + self.suffix_tensor_name,
              [self.feature_size, self.cluster_size],
              initializer = tf.random_normal_initializer(
              stddev=1 / math.sqrt(self.feature_size)))

        activation = tf.matmul(reshaped_input, cluster_weights)

        if self.add_batch_norm:
          activation = slim.batch_norm(
              activation,
              center=True,
              scale=True,
              is_training=self.is_training,
              scope="cluster_bn" + self.suffix_tensor_name)
        else:
          cluster_biases = tf.get_variable("cluster_biases" + self.suffix_tensor_name,
            [self.cluster_size],
            initializer = tf.random_normal_initializer(
            stddev=1 / math.sqrt(self.feature_size)))
          activation += cluster_biases

        activation = tf.nn.softmax(activation)

        activation = tf.reshape(activation,
                [-1, self.max_samples, self.cluster_size])

        a_sum = tf.reduce_sum(activation,-2,keep_dims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2" + self.suffix_tensor_name,
            [1,self.feature_size, self.cluster_size],
            initializer = tf.random_normal_initializer(
                stddev=1 / math.sqrt(self.feature_size)))

        a = tf.multiply(a_sum,cluster_weights2)

        activation = tf.transpose(activation,perm=[0,2,1])

        reshaped_input = tf.reshape(reshaped_input,[-1,
            self.max_samples, self.feature_size])

        vlad = tf.matmul(activation,reshaped_input)
        vlad = tf.transpose(vlad,perm=[0,2,1])
        vlad = tf.subtract(vlad,a)


        vlad = tf.nn.l2_normalize(vlad,1)

        vlad = tf.reshape(vlad,[-1, self.cluster_size*self.feature_size])
        vlad = tf.nn.l2_normalize(vlad,1)

        hidden1_weights = tf.get_variable("hidden1_weights" + self.suffix_tensor_name,
          [self.cluster_size*self.feature_size, self.output_dim],
          initializer=tf.random_normal_initializer(
          stddev=1 / math.sqrt(self.cluster_size)))

        vlad = tf.matmul(vlad, hidden1_weights)

        if self.gating:
            vlad = super(self.__class__, self).context_gating(vlad)

        return vlad
