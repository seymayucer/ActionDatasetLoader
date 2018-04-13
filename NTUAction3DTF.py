import tensorflow as tf
from os import listdir
from os.path import join
import numpy as np
import workspace_generals
import helpers
from tensorflow.contrib.data import Dataset, Iterator
from six.moves import cPickle
from helpers import logger

max_frame_num = 300
frame_size = 75
num_classes = 60


def _parse_function(filename):
    img = np.loadtxt(filename, dtype=float)
    img_len = len(img)
    # label = tf.one_hot(label, num_classes, dtype=tf.int32)

    if img_len < max_frame_num:
        mis_dim = max_frame_num - img_len
        zero_vec = np.zeros((mis_dim, img.shape[1]))
        img = np.vstack((img, zero_vec))

    return img, img_len


def read(data_dir, subject_id, split=False):
    print('Loading NTU 3D Data, data directory %s' % data_dir)
    filenames = []

    documents = [d for d in sorted(listdir(data_dir))]
    filenames.extend(documents)
    np.random.shuffle(filenames)

    test_files, test_labels = [], []
    train_files, train_labels = [], []

    for files in filenames:
        substr = 'P{0:03}'.format(subject_id)
        label = helpers.full_fname2_str(files, 'A')
        if substr in files:
            test_files.append(join(data_dir, files))
            test_labels.append(label)
        else:
            train_files.append(join(data_dir, files))
            train_labels.append(label)

    train_imgs = tf.constant(train_files)
    train_labels = tf.constant(train_labels)

    test_imgs = tf.constant(test_files)
    test_labels = tf.constant(test_labels)
    # create TensorFlow Dataset objects

    train_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
    train_data = train_data.map(
        lambda filename, label: tuple(tf.py_func(_parse_function, [filename], [tf.double, tf.int64])) + (
            tf.one_hot(label, num_classes, dtype=tf.int32),))
    train_data = train_data.batch(32)  # batched dataset creation

    test_data = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels))
    test_data = test_data.map(
        lambda filename, label: tuple(tf.py_func(_parse_function, [filename], [tf.double, tf.int64])) + (
            tf.one_hot(label, num_classes, dtype=tf.int32),))

    print('data is ready', type(train_data), train_data)
    if split:
        return train_data, test_data
