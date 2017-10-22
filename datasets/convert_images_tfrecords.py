# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Downloads and converts Tumblr data to TFRecords of TF-Example protos.

This module reads the photos of Tumblr data and creates two TFRecord datasets: 
one for train and one for validation. Each TFRecord dataset is comprised of a set 
of TF-Example protocol buffers, each of which contain a single image and label.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import numpy as np
import tensorflow as tf

from scipy.misc import imread, imresize
from datetime import datetime
from slim.nets import inception
from datasets import dataset_utils
from text_model.text_preprocessing import preprocess_one_df
from text_model.text_preprocessing import _load_embedding_weights_glove, _load_embedding_weights_word2vec

# The number of images in the validation set.
#_NUM_VALIDATION = 50

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# Number of words in a post
_POST_SIZE = 50

# Filename containing the train/valid split size
_TRAIN_VALID_FILENAME = 'train_valid_split.txt'


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir, photos_subdir='photos'):
  """Returns a list of filenames and inferred class names.

  Parameters:
    dataset_dir: A directory containing a subdirectory photos_subdir that 
      contains a set of subdirectories representing class names. 
      Each subdirectory should contain JPG encoded images.
    photos_subdir: A subdirectory of dataset_dir.

  Returns:
    A list of image file paths, relative to `dataset_dir/photos_subdir` and 
    the list of subdirectories, representing class names.
  """
  root = os.path.join(dataset_dir, photos_subdir)
  directories = []
  class_names = []
  for filename in os.listdir(root):
    path = os.path.join(root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      if filename != '.DS_Store':
        path = os.path.join(directory, filename)
        photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, photos_subdir, split_name, shard_id):
  output_filename = 'tumblr_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, photos_subdir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, 
                     tfrecords_subdir='tfrecords'):
  """Converts the given filenames to a TFRecords dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
    tfrecords_subdir: A subdirectory to save the TFRecords dataset
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session() as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, tfrecords_subdir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()
            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _convert_dataset_with_text(split_name, filenames, class_names_to_ids, dataset_dir, df_dict, 
                               tfrecords_subdir='tfrecords'):
  """Converts the given filenames to a TFRecords dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
    tfrecords_subdir: A subdirectory to save the TFRecords dataset
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session() as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, tfrecords_subdir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()
            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            # Get index of the post
            base = os.path.basename(filenames[i])
            index = (int)(os.path.splitext(base)[0])
            text_data = df_dict[class_name].iloc[index]['text_list']
            seq_len = df_dict[class_name].iloc[index]['text_len']
            post_id = df_dict[class_name].iloc[index]['id']
            date = df_dict[class_name].iloc[index]['date']
            day = datetime.strptime(date, '%Y-%m-%d %H:%M:%S GMT').weekday()
            # Convert to str, as only strings are accepted by tfexample
            #text_data = text_data.encode('ascii', 'ignore')

            example = dataset_utils.image_to_tfexample_with_text(
                image_data, b'jpg', height, width, text_data, seq_len, class_id, post_id, day)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir, photos_subdir='photos'):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
    photos_subdir: The subdirectory where the temporary files are stored.
  """
  #filename = _DATA_URL.split('/')[-1]
  #filepath = os.path.join(dataset_dir, filename)
  #tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, photos_subdir)
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir, photos_subdir='photos'):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, photos_subdir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def convert_images(dataset_dir, num_valid, photos_subdir='photos', tfrecords_subdir='tfrecords'):
  """Downloads the photos and convert them to TFRecords.

  Parameters:
    dataset_dir: The data directory.
    photos_subdir: The subdirectory where the photos are stored.
    tfrecords_subdir: The subdirectory to store the TFRecords files.
  """
  # Create the tfrecords_subdir if it doesn't exist
  if not tf.gfile.Exists(os.path.join(dataset_dir, tfrecords_subdir)):
    tf.gfile.MakeDirs(os.path.join(dataset_dir, tfrecords_subdir))

  if _dataset_exists(dataset_dir, photos_subdir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir, photos_subdir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[num_valid:]
  validation_filenames = photo_filenames[:num_valid]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir, tfrecords_subdir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir, tfrecords_subdir)

  # Write the train/validation split size
  train_valid_split = dict(zip(['train', 'validation'], [len(photo_filenames) - num_valid, num_valid]))
  train_valid_filename = os.path.join(dataset_dir, photos_subdir, _TRAIN_VALID_FILENAME)
  with tf.gfile.Open(train_valid_filename, 'w') as f:
    for split_name in train_valid_split:
      size = train_valid_split[split_name]
      f.write('%s:%d\n' % (split_name, size))

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir, photos_subdir)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the dataset!')

def convert_images_with_text(dataset_dir, num_valid, photos_subdir='photos', tfrecords_subdir='tfrecords'):
  """Downloads the photos and convert them to TFRecords.

  Parameters:
    dataset_dir: The data directory.
    photos_subdir: The subdirectory where the photos are stored.
    tfrecords_subdir: The subdirectory to store the TFRecords files.
  """
  # Create the tfrecords_subdir if it doesn't exist
  if not tf.gfile.Exists(os.path.join(dataset_dir, tfrecords_subdir)):
    tf.gfile.MakeDirs(os.path.join(dataset_dir, tfrecords_subdir))

  if _dataset_exists(dataset_dir, photos_subdir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir, photos_subdir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[num_valid:]
  validation_filenames = photo_filenames[:num_valid]

  # Load dataframes
  df_dict = dict()
  emb_name = 'glove'
  text_dir = 'text_model'
  emb_dir = 'embedding_weights'
  filename = 'glove.6B.50d.txt'
  if emb_name == 'word2vec':
      vocabulary, embedding = _load_embedding_weights_word2vec(text_dir, emb_dir, filename)
  else:
      vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)

  for emotion in class_names:
    df_dict[emotion] = preprocess_one_df(vocabulary, embedding, emotion, _POST_SIZE)

  # First, convert the training and validation sets.
  _convert_dataset_with_text('train', training_filenames, class_names_to_ids,
                   dataset_dir, df_dict, tfrecords_subdir)
  _convert_dataset_with_text('validation', validation_filenames, class_names_to_ids,
                   dataset_dir, df_dict, tfrecords_subdir)

  # Write the train/validation split size
  train_valid_split = dict(zip(['train', 'validation'], [len(photo_filenames) - num_valid, num_valid]))
  train_valid_filename = os.path.join(dataset_dir, photos_subdir, _TRAIN_VALID_FILENAME)
  with tf.gfile.Open(train_valid_filename, 'w') as f:
    for split_name in train_valid_split:
      size = train_valid_split[split_name]
      f.write('%s:%d\n' % (split_name, size))

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir, photos_subdir)

  #_clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the dataset!')

def _filenames_to_arrays(filenames, class_names_to_ids):
    X = []
    y = []
    for filename in filenames:
        im = imread(filename)
        class_name = os.path.basename(os.path.dirname(filename))
        class_id = class_names_to_ids[class_name]
        # Might need different size as 224 is quite large
        image_size = 16#inception.inception_v1.default_image_size
        # Resize and flatten the image
        im = imresize(im, (image_size, image_size)).reshape(-1)
        X.append(im)
        y.append(class_id)
    X = np.vstack(X)
    y = np.array(y)
    return (X, y)

def get_numpy_data(dataset_dir, num_valid, photos_subdir='photos'):
  """Convert the photos to numpy data.

  Parameters:
    dataset_dir: The data directory.
    photos_subdir: The subdirectory where the photos are stored.

  Returns:
    X_train: Training features.
    X_valid: Validation features.
    y_train: Training labels.
    y_valid: Validation labels
  """
  photo_filenames, class_names = _get_filenames_and_classes(dataset_dir, photos_subdir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[num_valid:]
  validation_filenames = photo_filenames[:num_valid]

  X_train, y_train = _filenames_to_arrays(training_filenames, class_names_to_ids)
  X_valid, y_valid = _filenames_to_arrays(validation_filenames, class_names_to_ids)

  return (X_train, X_valid, y_train, y_valid)
