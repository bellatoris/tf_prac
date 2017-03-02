from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time

import ResNet
from cifar_input import build_input


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                           [32, 32, 3]))
    labels_placeholder = tf.placeholder(tf.int32, shape=batch_size)
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set

    feed_dict = {
        images_pl: images_feed,
        labels_pl: labels_feed
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
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = float(true_count) / num_examples
    print(' Num examples: %d Num correct: %d Precision @ 1: %0.04f' %
          (num_examples, true_count, precision))


def run_training():
    data_sets = build_input('./data', 128, 'training')
    test_sets = build_input('./data', 100, 'test')

    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(128)
        logits = ResNet.resnet(images_placeholder)
        loss = ResNet.loss(logits, labels_placeholder)
        train_op = ResNet.training(loss, 0.1)
        eval_correct = ResNet.evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess = tf.Session()
        summary_writer = tf.summary.FileWriter('./logs', sess.graph)
        sess.run(init)

        for step in range(64000):
            start_time = time.time()

            feed_dict = fill_feed_dict(data_sets,
                                       images_placeholder,
                                       labels_placeholder)

            _, loss_value = sess.run([train_op, loss],
                                     feed_dict=feed_dict)

            duration = time.time() - start_time

            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            if (step + 1) % 1000 == 0 or (step + 1) == 64000:
                # Evaluate against the training set.
                print('Training Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets)
                # Evaluate against the test set.
                print('Test Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        test_sets)


def main(_):
    run_training()

