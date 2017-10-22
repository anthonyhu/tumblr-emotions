# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides data for the Tumblr dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'tumblr_%s_*.tfrecord'

# Filename containing the train/valid split size
_TRAIN_VALID_FILENAME = 'train_valid_split.txt'

_POST_SIZE = 50

#SPLITS_TO_SIZES = {'train': 128, 'validation': 30}

#_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and (num_classes - 1)',
}


def get_split(split_name, dataset_dir, photos_subdir='photos', tfrecords_subdir='tfrecords', 
              file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading tumblr data.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    photos_subdir: The subdirectory containing the photos.
    tfrecords_subdir: The subdirectory containing the TFRecords files.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  #if split_name not in SPLITS_TO_SIZES:
    #raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, tfrecords_subdir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir, photos_subdir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir, photos_subdir)

  # Get split size
  train_valid_filename = os.path.join(dataset_dir, photos_subdir, _TRAIN_VALID_FILENAME)
  with tf.gfile.Open(train_valid_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  train_valid_split = {}
  for line in lines:
    index = line.index(':')
    train_valid_split[line[:index]] = (int)(line[index+1:])

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=train_valid_split[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=len(labels_to_names),
      labels_to_names=labels_to_names)

def get_split_with_text(split_name, dataset_dir, photos_subdir='photos', tfrecords_subdir='tfrecords', 
              file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading tumblr data.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    photos_subdir: The subdirectory containing the photos.
    tfrecords_subdir: The subdirectory containing the TFRecords files.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  #if split_name not in SPLITS_TO_SIZES:
    #raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  file_pattern = os.path.join(dataset_dir, tfrecords_subdir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'text': tf.FixedLenFeature(
          [_POST_SIZE], tf.int64, default_value=tf.zeros([_POST_SIZE], dtype=tf.int64)),
      'seq_len': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'post_id': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
      'day': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
  }

  items_to_handlers = {
      'image': slim.tfexample_decoder.Image(),
      'text': slim.tfexample_decoder.Tensor('text'),
      'seq_len': slim.tfexample_decoder.Tensor('seq_len'),
      'label': slim.tfexample_decoder.Tensor('image/class/label'),
      'post_id': slim.tfexample_decoder.Tensor('post_id'),
      'day': slim.tfexample_decoder.Tensor('day'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
      keys_to_features, items_to_handlers)

  labels_to_names = None
  if dataset_utils.has_labels(dataset_dir, photos_subdir):
    labels_to_names = dataset_utils.read_label_file(dataset_dir, photos_subdir)

  # Get split size
  train_valid_filename = os.path.join(dataset_dir, photos_subdir, _TRAIN_VALID_FILENAME)
  with tf.gfile.Open(train_valid_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  train_valid_split = {}
  for line in lines:
    index = line.index(':')
    train_valid_split[line[:index]] = (int)(line[index+1:])

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=train_valid_split[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      num_classes=len(labels_to_names),
      labels_to_names=labels_to_names)
