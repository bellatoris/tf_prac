from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base


# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
    """Reads and parses examples from CIFAR10 data files.

    :param filename_queue: A queue of strings with the filenames to read from.
    :return:
        An object representing a single example, with the following fields:
        height: number of rows in the result (32)
        width: number of columns in the result (32)
        depth: number of color channel in the result (3)
        key: a scalar string Tensor describing the filename & record number
            for this example.
        label: an int32 Tensor with the label in the range 0..9.
        uint8image: a [height, width, depth] uint8 Tensor with the image data.
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8 -> int32.
    result.label = tf.cast(
        tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32
    )

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes],
                         [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])
    # Convert from [depth, height, width] to [height, width, depth].
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """ Construct a queued batch of images and labels.

    :param image: 3-D Tensor of [height, width, 3] of type.float32
    :param label: 1-D Tensor of type.int32
    :param min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
    :param batch_size: number of images per batch.
    :param shuffle: boolean indicating whether to use a shuffling queue.
    :return:
        images: Images. 4D Tensor of [batch_size, height, width, 3] size.
        labels: Labels. 1D Tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size
        )

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images. tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

    :param data_dir: Path to the CIFAR-10 data directory.
    :param batch_size: number of images per batch.
    :return:
        images: Images. 4D Tensor of [batch_size, 32, 32, 3] size.
        labels: Labels. 1D Tensor of [batch_size] size.
    """
    SOURCE_URL = 'https://www.cs.toronto.edu/~kriz/'
    CIFAR10 = 'cifar-10-binary.tar.gz'
    DATASET = 'cifar-10-batches-bin'
    ROOT = os.path.join(data_dir, DATASET)

    local_file = base.maybe_download(CIFAR10, data_dir, SOURCE_URL + CIFAR10)
    if not os.path.exists(ROOT):
        with tarfile.open(local_file, "r:gz") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, data_dir)
            tar.close()

    filenames = [os.path.join(ROOT, 'data_batch_%d.bin' % i)
                 for i in range(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    height = read_input.height
    width = read_input.width
    depth = read_input.depth
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    image = tf.image.resize_image_with_crop_or_pad(
        reshaped_image, height + 4, width + 4)
    image = tf.random_crop(image, [height, width, depth])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.per_image_standardization(image)

    # Set the shapes of tensors.
    image.set_shape([height, width, depth])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evaluation using the Reader ops.

    :param data_dir: Path to the CIFAR-10 data directory.
    :param batch_size: number of images per batch.
    :return:
        images: Images. 4D Tensor of [batch_size, 32, 32, 3] size.
        labels: Labels. 1D Tensor of [batch_size] size.
    """
    DATASET = 'cifar-10-batches-bin'
    ROOT = os.path.join(data_dir, DATASET)
    if not eval_data:
        filenames = [os.path.join(ROOT, 'data_batch_%d.bin' % i)
                     for i in range(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(ROOT, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    # Read examples from files in the filename queue.
    read_input = read_cifar10(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = read_input.height
    width = read_input.width
    depth = read_input.depth

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    image.set_shape([height, width, 3])
    read_input.label.se_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)

