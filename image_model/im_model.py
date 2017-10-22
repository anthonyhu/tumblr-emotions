""" Fine-tune a pre-trained Inception model by chopping off the last logits layer. 
"""
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.learning import train_step
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from slim.preprocessing import inception_preprocessing
#from slim.nets import inception
from image_model import inception_v1
from datasets import dataset_utils
from datasets.convert_to_dataset import get_split, get_split_with_text
from datasets.convert_images_tfrecords import get_numpy_data

# Seed for reproducibility
_RANDOM_SEED = 0
_CONFIG = {'mode': 'train',
           'dataset_dir': 'data',
           'initial_lr': 1e-3,
           'decay_factor': 0.3,
           'batch_size': 64,
           'final_endpoint': 'Mixed_5c'}

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
    """Load a single batch of data. 
    
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

def load_batch_with_text(dataset, batch_size=32, shuffle=True, height=299, width=299, is_training=False):
    """Load a single batch of data. 
    
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
    image_raw, text, seq_len, label, post_id, day = data_provider.get(['image', 'text', 'seq_len', 'label', 'post_id', 'day'])
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, texts, seq_lens, labels, post_ids, days = tf.train.batch(
        [image, image_raw, text, seq_len, label, post_id, day],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)
    
    return images, images_raw, texts, seq_lens, labels, post_ids, days 

def get_init_fn(checkpoints_dir, model_name='inception_v1.ckpt'):
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

class ImageModel():
    def __init__(self, config):
        self.config = config
        mode = config['mode']
        dataset_dir = config['dataset_dir']
        initial_lr = config['initial_lr']
        batch_size = config['batch_size']
        final_endpoint = config['final_endpoint']

        tf.logging.set_verbosity(tf.logging.INFO)

        self.learning_rate = tf.Variable(initial_lr, trainable=False)
        self.lr_rate_placeholder = tf.placeholder(tf.float32)
        self.lr_rate_assign = self.learning_rate.assign(self.lr_rate_placeholder)

        self.dataset = get_split_with_text(mode, dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, seq_lens, self.labels = load_batch_with_text(self.dataset, batch_size, height=image_size, width=image_size)
            
        self.nb_emotions = self.dataset.num_classes
        # Create the model, use the default arg scope to configure the batch norm parameters.
        is_training = (mode == 'train')
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            self.logits, _ = inception_v1.inception_v1(images, final_endpoint=final_endpoint,
                num_classes=self.nb_emotions, is_training=is_training)

def train_image_model(checkpoints_dir, train_dir, num_steps):
    """Fine tune the Image model, retraining Mixed_5c.

    Parameters:
        checkpoints_dir: The directory contained the pre-trained model.
        train_dir: The directory to save the trained model.
        num_steps: The number of steps training the model.
    """
    if tf.gfile.Exists(train_dir):
        # Delete old model
        tf.gfile.DeleteRecursively(train_dir)
    tf.gfile.MakeDirs(train_dir)

    with tf.Graph().as_default():
        model = ImageModel(_CONFIG)
        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(model.labels, model.nb_emotions)
        slim.losses.softmax_cross_entropy(model.logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process
        # Use tensorboard --logdir=train_dir, careful with path (add Documents/tumblr-sentiment in front of train_dir)
        # Different from the logs, because computed on different mini batch of data
        tf.summary.scalar('Loss', total_loss)
      
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=model.learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        batch_size = _CONFIG['batch_size']
        initial_lr = _CONFIG['initial_lr']
        decay_factor = _CONFIG['decay_factor']
        nb_batches = model.dataset.num_samples / batch_size
        def train_step_fn(session, *args, **kwargs):
            # Decaying learning rate every epoch
            if train_step_fn.step % (nb_batches) == 0:
                lr_decay = decay_factor ** train_step_fn.epoch
                session.run(model.lr_rate_assign, feed_dict={model.lr_rate_placeholder: initial_lr * lr_decay})
                print('New learning rate: {0}'. format(initial_lr * lr_decay))
                train_step_fn.epoch += 1

            total_loss, should_stop = train_step(session, *args, **kwargs)

            #variables_to_print = ['InceptionV1/Conv2d_2b_1x1/weights:0', 'InceptionV1/Mixed_4b/Branch_3/Conv2d_0b_1x1/weights:0',
             #                     'InceptionV1/Logits/Conv2d_0c_1x1/weights:0']
            #for v in slim.get_model_variables():
             #   if v.name in variables_to_print:
              #      print(v.name)
               #     print(session.run(v))
                #    print('\n')
            #acc_valid = session.run(accuracy_valid)
            #print('Step {0}: loss: {1:.3f}, validation accuracy: {2:.3f}'.format(train_step_fn.step, total_loss, acc_valid))
            #sys.stdout.flush()
            train_step_fn.step += 1
            return [total_loss, should_stop]
        
        train_step_fn.step = 0
        train_step_fn.epoch = 0

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            init_fn=get_init_fn(checkpoints_dir),
            save_interval_secs=600,
            save_summaries_secs=600,
            train_step_fn=train_step_fn,
            number_of_steps=num_steps)
            
    print('Finished training. Last batch loss {0:.3f}'.format(final_loss))

def evaluate_image_model(checkpoint_dir, log_dir, mode, num_evals):
    """Visualise results with: tensorboard --logdir=logdir. Now has train/validation curves on the same plot
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
        log_dir: Directory to save logs.
        mode: train or validation.
        num_evals: Number of batches to evaluate (mean of the batches is displayed).
    """
    with tf.Graph().as_default():
        _CONFIG['mode'] = mode
        model = ImageModel(_CONFIG)

        # Accuracy metrics
        accuracy = slim.metrics.streaming_accuracy(tf.cast(model.labels, tf.int32),
                                                   tf.cast(tf.argmax(model.logits, 1), tf.int32))

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': accuracy,
        })

        for metric_name, metric_value in names_to_values.iteritems():
            tf.summary.scalar(metric_name, metric_value)

        log_dir = os.path.join(log_dir, mode)

        # Evaluate every eval_interval_secs secs or if not specified,
        # every time the checkpoint_dir changes
        # tf.get_variable variables are also restored
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            log_dir,
            num_evals=num_evals,
            eval_op=names_to_updates.values())
        
def softmax_regression(num_valid, C):
    """Run a softmax regression on the images.

    Parameters:
        num_valid: Size of the validation set.
        C: Inverse of the regularization strength.
    """
    # Load data
    X_train, X_valid, y_train, y_valid = get_numpy_data('data', num_valid)
    logistic = LogisticRegression(multi_class='multinomial', solver='newton-cg',
                                  C=C, random_state=_RANDOM_SEED)
    print('Start training Logistic Regression.')
    logistic.fit(X_train, y_train)

    accuracy_train = accuracy_score(logistic.predict(X_train), y_train)
    valid_accuracy = accuracy_score(logistic.predict(X_valid), y_valid)
    print('Training accuracy: {0:.3f}'.format(accuracy_train))
    print('Validation accuracy: {0:.3f}'.format(valid_accuracy))

def forest(num_valid, n_estimators, max_depth):
    """Run a Random Forest on the images.

    Parameters:
        num_valid: Size of the validation set.
        n_estimators: Number of trees.
        max_depth: Maximum depth of a tree.
    """
    # Load data
    X_train, X_valid, y_train, y_valid = get_numpy_data('data', num_valid)
    forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                    random_state=_RANDOM_SEED)
    print('Start training Random Forest.')
    forest.fit(X_train, y_train)

    accuracy_train = accuracy_score(forest.predict(X_train), y_train)
    valid_accuracy = accuracy_score(forest.predict(X_valid), y_valid)
    print('Training accuracy: {0:.3f}'.format(accuracy_train))
    print('Validation accuracy: {0:.3f}'.format(valid_accuracy))