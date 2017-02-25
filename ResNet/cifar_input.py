from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base


def build_input(data_path, batch_size, mode):
    """Build CIFAR image and labels

    Args:
        data_path: Filename for data
        batch_size: Input batch size.
        mode: Either 'train' or 'eval'.
    Returns:
        images: Batches of images. [batch_size, image_size, image_size, 3]
        labels: Batches of labels. [batch_size, num_classes]
    """
    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
    CIFAR10 = 'cifar-10-binary.tar.gz'
    DATASET = 'cifar-10-batches-bin'
    ROOT = os.path.join(data_path, DATASET)

    local_file = base.maybe_download(CIFAR10, data_path, SOURCE_URL + CIFAR10)
    if not os.path.exists(ROOT):
        with tarfile.open(local_file, "r:gz") as tar:
            tar.extractall(data_path)
            tar.close()

    image_size = 32
    label_bytes = 1
    num_classes = 10
    channel = 3

    if mode == "train":
        filenames = [os.path.join(ROOT, 'data_batch_%d.bin' % i)
                     for i in range(1, 6)]
    else:
        filenames = [os.path.join(ROOT, 'test_batch.bin')]

    # Create a queue that produces the filenames to read.
    file_queue = tf.train.string_input_producer(filenames, shuffle=True)

    image_bytes = image_size * image_size * channel
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue. No
    # header or footer in the CIFAR-10 format, so we leave header_bytes
    # and footer_bytes at their default of 0.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    _, value = reader.read(file_queue)

    # Convert from a sting to vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [channel * height* width] to [channel, height, width]
    channel_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [channel, image_size, image_size])
    # Convert from [channel, height, width] to [height, width, channel]
    image = tf.cast(tf.transpose(channel_major, [1, 2, 0]), tf.float32)

    if mode == 'train':
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size + 4, image_size + 4)
        image = tf.random_crop(image, [image_size, image_size, channel])
        image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)

        example_queue = tf.RandomShuffleQueue(
            capacity=16 * batch_size,
            min_after_dequeue=8 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, channel], [1]])
        num_threads = 16
    else:
        image = tf.image.per_image_standardization(image)

        example_queue = tf.FIFOQueue(
            3 * batch_size,
            dtypes=[tf.float32, tf.int32],
            shapes=[[image_size, image_size, channel], [1]])
        num_threads = 1

    example_queue_op = example_queue.enqueue([image, label])
    tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
        example_queue, [example_queue_op] * num_threads))

    # Read 'batch' labels + images from the example queue
    images, labels = example_queue.dequeue_many(batch_size)
    labels = tf.reshape(labels, [batch_size, 1])
    # indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])
    # labels = tf.sparse_to_dense(
    #     tf.concat(values=[indices, labels], axis=1),
    #     [batch_size, num_classes], 1.0, 0.0)

    # assert len(images.get_shape()) == 4
    # assert images.get_shape()[0] == batch_size
    # assert images.get_shape()[-1] == 3
    # assert len(labels.get_shape()) == 2
    # assert labels.get_shape()[0] == batch_size
    # assert labels.get_shape()[1] == num_classes

    # Display the training images in the visualizer
    tf.summary.image('images', images)
    return images, labels
