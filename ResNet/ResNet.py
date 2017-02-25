from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


NUM_CLASSES = 10
IMAGE_SIZE = 32
CHANNEL = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * CHANNEL

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W, stride=1):
  return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def shortcut(x, input_channel, output_channel, stride):
    W_conv1 = weight_variable([1, 3, 3, 1])
    h_conv1 = tf.nn.relu(tf.nn.batch_normalization(conv2d(x, W_conv1)))


def inference(images):


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