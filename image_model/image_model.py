""" Fine-tune a pre-trained Inception model by chopping off the last logits layer. 
"""
import os

import tensorflow as tf

from datasets import dataset_utils
from preprocessing import inception_preprocessing
from slim.nets import inception
#from datasets import flowers

from tensorflow.contrib import slim

# Inception image size
_IMAGE_SIZE = inception.inception_v1.default_image_size

def download_pretrained_model(url, checkpoint_dir):
	"""Download pretrained inception model and store it in checkpoint_dir.

	Parameters:
		url: The url containing the compressed model.
		checkpoint_dir: The directory to save the model.
	"""
	if not tf.gfile.Exists(checkpoint_dir):
		tf.gfile.MakeDirs(checkpoint_dir)
	dataset_utils.download_and_uncompress_tarball(url, checkpoint_dir)

def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return images, images_raw, labels