from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from conv_block import softmax_layer, conv_layer, residual_block

NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNEL = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNEL

def resnet(inpt, n=110):
    num_conv = (n - 20) / 12 + 1
    layers = []

    with tf.variable_scope('conv1'):
        conv1 = conv_layer(inpt, [3, 3, 3, 16], 1)
        layers.append(conv1)

    for i in range(num_conv):
        with tf.variable_scope('conv2_%d' % (i+1)):
            conv2_x = residual_block(layers[-1], 16, False)
            conv2 = residual_block(conv2_x, 16, False)
            layers.append(conv2_x)
            layers.append(conv2)

        assert conv2.get_shape.as_list()[1:] == [32, 32, 16]

    for i in range(num_conv):
        with tf.variable_scope('conv3_%d' % (i + 1)):
            conv3_x = residual_block(layers[-1], 32, False)
            conv3 = residual_block(conv3_x, 32, False)
            layers.append(conv3_x)
            layers.append(conv3)

        assert conv2.get_shape.as_list()[1:] == [16, 16, 32]


def inference(images):
    i = 3


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