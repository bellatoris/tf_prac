from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


# initialize weight variable randomly with specific standard deviation.
from tensorflow.python.training import moving_averages


def weight_variable(shape, name):
    initial = tf.constant(1.0, shape=shape, name=name)
    return tf.Variable(initial, name=name)


def weight_variable_with_decay(shape, name, weight_decay):
    n = shape[0] * shape[1] * shape[2]
    var = tf.get_variable(name, shape,
                          initializer=tf.random_normal_initializer(
                              stddev=np.sqrt(2.0/n)))
    weight_loss = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
    tf.add_to_collection('losses', weight_loss)
    return var


def bias_variable(shape, name=None):
    initial = tf.constant(0, shape=shape)
    return tf.Variable(initial, name=name)


def fc_layer(inpt, shape, weight_decay):
    fc_w = weight_variable_with_decay(shape, 'fc', weight_decay)
    fc_b = bias_variable(shape[1])
    fc_h = tf.matmul(inpt, fc_w) + fc_b

    return fc_h


def conv2d(x, filter_, stride=1):
    return tf.nn.conv2d(x, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')


# Convolution layer followed by batch normalization and ReLU.
def conv_layer(inpt, filter_shape, name, stride, weight_decay, train):
    out_channels = filter_shape[3]

    filter_ = weight_variable_with_decay(filter_shape, name, weight_decay)
    conv = conv2d(inpt, filter_, stride)

    # after convolution
    beta = bias_variable([out_channels], name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    if train:
        mean, var = tf.nn.moments(conv, axes=[0, 1, 2])

        moving_mean = tf.get_variable(
            'moving_mean', [out_channels], tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False
        )
        moving_variance = tf.get_variable(
            'moving_variance', [out_channels], tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False
        )
        moving_averages.assign_moving_average(
            moving_mean, mean, 0.9)
        moving_averages.assign_moving_average(
            moving_variance, var, 0.9)
    else:
        mean = tf.get_variable(
            'moving_mean', [out_channels], tf.float32,
            initializer=tf.constant_initializer(0.0, tf.float32),
            trainable=False
        )
        var = tf.get_variable(
            'moving_variance', [out_channels], tf.float32,
            initializer=tf.constant_initializer(1.0, tf.float32),
            trainable=False
        )
    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True
    )

    return batch_norm


# basic residual block. It isn't bottleneck structure.
def residual_block(inpt, output_depth, weight_decay, projection=True, train=True):
    input_depth = inpt.get_shape()[3].value
    if input_depth != output_depth:
        stride = 2
        if projection:
            # Option B: Projection shortcut
            shortcut = conv_layer(inpt, [1, 1, input_depth, output_depth],
                                  name='shortcut', stride=2,
                                  weight_decay=weight_decay, train=train)
        else:
            # Option A: Zero-padding
            shortcut = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        stride = 1
        shortcut = inpt

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth],
                       name='conv1', stride=stride,
                       weight_decay=weight_decay, train=train)
    conv2 = conv_layer(tf.nn.relu(conv1), [3, 3, output_depth, output_depth],
                       name='conv2', stride=1,
                       weight_decay=weight_decay, train=train)

    res = tf.nn.relu(conv2) + shortcut
    return res

