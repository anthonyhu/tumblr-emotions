""" Fine-tune a pre-trained Inception model by chopping off the last logits layer. 
"""
import os
import sys

import numpy as np
import tensorflow as tf

from datasets import dataset_utils
from datasets.convert_to_dataset import get_split
from slim.preprocessing import inception_preprocessing
from slim.nets import inception
from tensorflow.contrib.slim.python.slim.learning import train_step

from tensorflow.contrib import slim

def download_pretrained_model(url, checkpoint_dir):
    """Download pretrained inception model and store it in checkpoint_dir.

    Parameters:
        url: The url containing the compressed model.
        checkpoint_dir: The directory to save the model.
    """
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)
    dataset_utils.download_and_uncompress_tarball(url, checkpoint_dir)

def _load_batch(dataset, batch_size=32, shuffle=True, height=299, width=299, is_training=False):
    """Loads a single batch of data. 
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      shuffle: Whether to shuffle the data sources and common queue when reading.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    # For validation, if you set the common_queue_capacity to something lower than
    # batch_size, which is the validation size, then your output will contain duplicates.
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, shuffle=shuffle, common_queue_capacity=batch_size,
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

def _get_init_fn(checkpoints_dir, model_name='inception_v1.ckpt'):
    """Returns a function run by the chief worker to warm-start the training.
    """
    checkpoint_exclude_scopes=["InceptionV1/Logits", "InceptionV1/AuxLogits"]
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, model_name),
        variables_to_restore)

def fine_tune_model(dataset_dir, checkpoints_dir, train_dir, num_steps):
    """Fine tune the inception model, retraining the last layer.

    Parameters:
        dataset_dir: The directory containing the data.
        checkpoints_dir: The directory contained the pre-trained model.
        train_dir: The directory to save the trained model.
        num_steps: The number of steps training the model.
    """
    if tf.gfile.Exists(train_dir):
        # Delete old model
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)
        
        dataset = get_split('train', dataset_dir)
        image_size = inception.inception_v1.default_image_size
        images, _, labels = _load_batch(dataset, height=image_size, width=image_size)

        # Load validation data
        dataset_valid = get_split('validation', dataset_dir)
        images_valid, _, labels_valid = _load_batch(dataset_valid, batch_size=dataset_valid.num_samples, shuffle=False, 
                                                    height=image_size, width=image_size)
        
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception.inception_v1_arg_scope()):
            logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)
            logits_valid, _ = inception.inception_v1(images_valid, num_classes=dataset_valid.num_classes, is_training=False)
            
        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)
      
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Accuracy metrics
        accuracy_valid = slim.metrics.accuracy(tf.cast(labels_valid, tf.int32),
                                               tf.cast(tf.argmax(logits_valid, 1), tf.int32))

        def train_step_fn(session, *args, **kwargs):
            total_loss, should_stop = train_step(session, *args, **kwargs)

            acc_valid = session.run(accuracy_valid)
            print('Step {0}: loss: {1:.3f}, validation accuracy: {2:.3f}'.format(train_step_fn.step, total_loss, acc_valid))
            sys.stdout.flush()
            train_step_fn.step += 1
            return [total_loss, should_stop]
        
        train_step_fn.step = 0

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            init_fn=_get_init_fn(checkpoints_dir),
            train_step_fn=train_step_fn,
            number_of_steps=num_steps)
            
    print('Finished training. Last batch loss {0:.3f}'.format(final_loss))
