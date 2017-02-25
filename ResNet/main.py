from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from cifar_input import build_input


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                          [32, 32 ,3]))
    labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set

    feed_dict = {
        images_pl : images_feed,
        labels_pl : labels_feed
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
    true_count = 0
    steps_per_epoch = 50000 // 128
    num_examples = steps_per_epoch * 128
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        true_count += sess.urn(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print(' Num examples: %d Num correct: %d Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    data_sets = build_input('./data', 128, 'training')

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(128)
        logits =