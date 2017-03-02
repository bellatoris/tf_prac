from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from conv_block import fc_layer, conv_layer, residual_block

NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNEL = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNEL


# Make ResNet model
def resnet(inpt, n=110):
    num_conv = (n - 2) / 3
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)

    for i in range(num_conv):
        with tf.variable_scope('conv2_%d' % i):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape.as_list()[1:] == [32, 32, 16]

    for i in range(num_conv):
        with tf.variable_scope('conv3_%d' % i):
            conv3_x = residual_block(layers[-1], 32, False)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv3.get_shape.as_list()[1:] == [16, 16, 32]

    for i in range(num_conv):
        with tf.variable_scope('conv4_%d' % i):
            conv4_x = residual_block(layers[-1], 64, False)
            conv4 = residual_block(conv4_x, 64, False)
            layers.append(conv3_4)
            layers.append(conv4)

        assert conv4.get_shape.as_list()[1:] == [8, 8, 64]

    with tf.variable_scope('fc'):
        global_pool = tf.reduce_mean(layers[-1], [1, 2])
        assert global_pool.get_shape().as_list()[1:] == [64]

        out = fc_layer(global_pool, [64, 10])
        layers.append(out)

    return layers[-1]


def loss(logits, labels):
    labels = tf.to_int64(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=0.9, use_nesterov=True)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))