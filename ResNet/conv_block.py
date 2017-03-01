from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# initialize weight variable randomly with specific standard deviation.
def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name=None):
    initial = tf.constant(0, shape=shape)
    return tf.Variable(initial, name=name)


def softmax_layer(inpt, shape):
    fc_w = weight_variable(shape)
    fc_b = bias_variable(shape[1])
    fc_h = tf.nn.softmax(tf.matmul(inpt, fc_w) + fc_b)

    return fc_h


def conv2d(x, filter_, stride=1):
    return tf.nn.conv2d(x, filter=filter_, strides=[1, stride, stride, 1], padding='SAME')


# Convolution layer followed by batch normalization and ReLU.
def conv_layer(inpt, filter_shape, stride):
    out_channels = filter_shape[3]

    filter_ = weight_variable(filter_shape)
    conv = conv2d(inpt, filter_, stride)
    mean, var = tf.nn.moments(conv, axes=[0,1,2])
    beta = bias_variable([out_channels], name="beta")
    gamma = weight_variable([out_channels], name="gamma")

    batch_norm = tf.nn.batch_norm_with_global_normalization(
        conv, mean, var, beta, gamma, 0.001,
        scale_after_normalization=True
    )

    out = tf.nn.relu(batch_norm)

    return out


# basic residual block. It isn't bottleneck structure.
def residual_block(inpt, output_depth,  projection=True):
    input_depth = inpt.get_shape()[3].value
    if input_depth != output_depth:
        stride = 2
        if projection:
            # Option B: Projection shortcut
            shortcut = conv_layer(inpt, [1, 1, input_depth, output_depth], stride=2)
        else:
            # Option A: Zero-padding
            shortcut = tf.pad(inpt, [[0, 0], [0, 0], [0, 0], [0, output_depth - input_depth]])
    else:
        stride = 1
        shortcut = inpt

    conv1 = conv_layer(inpt, [3, 3, input_depth, output_depth], stride=stride)
    conv2 = conv_layer(conv1, [3, 3, output_depth, output_depth], stride=1)

    res = conv2 + shortcut
    return res

