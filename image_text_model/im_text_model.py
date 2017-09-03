import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.learning import train_step

from slim.preprocessing import inception_preprocessing
from image_model import inception_v1
from datasets import dataset_utils
from text_model.text_preprocessing import _load_embedding_weights_glove
from image_model.im_model import load_batch_with_text, get_init_fn
from datasets.convert_to_dataset import get_split_with_text

_POST_SIZE = 200

def train_deep_sentiment(dataset_dir, checkpoints_dir, train_dir, num_steps, learning_rate):
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
        
        dataset = get_split_with_text('train', dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, labels = load_batch_with_text(dataset, height=image_size, width=image_size)
        
        fc1_size = 2048
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, num_classes=fc1_size, is_training=True)

        # Text model
        text_dir = 'text_model'
        emb_dir = 'embedding_weights'
        filename = 'glove.6B.50d.txt'
        vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
        vocab_size, embedding_dim = embedding.shape
        word_to_id = dict(zip(vocabulary, range(vocab_size)))
        # Unknown words = vector with zeros
        embedding = np.concatenate([embedding, np.zeros((1, embedding_dim))])
        word_to_id['<ukn>'] = vocab_size

        vocab_size = len(word_to_id)
        nb_emotions = dataset.num_classes
        with tf.variable_scope('Text'):
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        
            # Word embedding
            W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
            embedding_init = W_embedding.assign(embedding_placeholder)
            input_embed = tf.nn.embedding_lookup(W_embedding, texts)
            #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

            # Rescale the mean by the actual number of non-zero values.
            nb_finite = tf.reduce_sum(tf.cast(tf.not_equal(input_embed, 0.0), tf.float32), axis=1)
            # If a post has zero finite elements, replace nb_finite by 1
            nb_finite = tf.where(tf.equal(nb_finite, 0.0), tf.ones_like(nb_finite), nb_finite)
            h1 = tf.reduce_mean(input_embed, axis=1) * _POST_SIZE / nb_finite

            # Fully connected layer
            W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
            b_fc1 = tf.get_variable('b_fc1', [fc1_size])
            texts_features = tf.matmul(h1, W_fc1) + b_fc1
            texts_features = tf.nn.relu(texts_features)

        # Stack image and text features
        stacked_features = tf.stack([images_features, texts_features], axis=1)
        stacked_features = tf.reshape(stacked_features, [-1, 2 * fc1_size])

        W_softmax = tf.get_variable('W_softmax', [2 * fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(stacked_features, W_softmax) + b_softmax
        # Specify the loss function:
        one_hot_labels = slim.one_hot_encoding(labels, nb_emotions)
        slim.losses.softmax_cross_entropy(logits, one_hot_labels)
        total_loss = slim.losses.get_total_loss()

        # Create some summaries to visualize the training process
        # Use tensorboard --logdir=train_dir, careful with path (add Documents/tumblr-sentiment in front of train_dir)
        # Different from the logs, because computed on different mini batch of data
        tf.summary.scalar('Loss', total_loss)
      
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        def train_step_fn(session, *args, **kwargs):
            # Initialise embedding weights
            if train_step_fn.step == 0:
                session.run(embedding_init, feed_dict={embedding_placeholder: embedding})
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

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            init_fn=get_init_fn(checkpoints_dir),
            save_interval_secs=60,
            save_summaries_secs=60,
            train_step_fn=train_step_fn,
            number_of_steps=num_steps)
            
    print('Finished training. Last batch loss {0:.3f}'.format(final_loss))

def evaluate_deep_sentiment(checkpoint_dir, log_dir, mode, num_evals):
    """Visualise results with: tensorboard --logdir=logdir. Now has train/validation curves on the same plot
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
        log_dir: Directory to save logs.
        mode: train or validation.
        num_evals: Number of batches to evaluate (mean of the batches is displayed).
    """
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        dataset_dir = 'data'
        dataset = get_split_with_text('train', dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, labels = load_batch_with_text(dataset, height=image_size, width=image_size)
        
        fc1_size = 2048
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, num_classes=fc1_size, is_training=True)

        # Text model
        text_dir = 'text_model'
        emb_dir = 'embedding_weights'
        filename = 'glove.6B.50d.txt'
        vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
        vocab_size, embedding_dim = embedding.shape
        word_to_id = dict(zip(vocabulary, range(vocab_size)))
        # Unknown words = vector with zeros
        embedding = np.concatenate([embedding, np.zeros((1, embedding_dim))])
        word_to_id['<ukn>'] = vocab_size

        vocab_size = len(word_to_id)
        nb_emotions = dataset.num_classes
        with tf.variable_scope('Text'):
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        
            # Word embedding
            W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
            embedding_init = W_embedding.assign(embedding_placeholder)
            input_embed = tf.nn.embedding_lookup(W_embedding, texts)
            #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

            # Rescale the mean by the actual number of non-zero values.
            nb_finite = tf.reduce_sum(tf.cast(tf.not_equal(input_embed, 0.0), tf.float32), axis=1)
            # If a post has zero finite elements, replace nb_finite by 1
            nb_finite = tf.where(tf.equal(nb_finite, 0.0), tf.ones_like(nb_finite), nb_finite)
            h1 = tf.reduce_mean(input_embed, axis=1) * _POST_SIZE / nb_finite

            # Fully connected layer
            W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
            b_fc1 = tf.get_variable('b_fc1', [fc1_size])
            texts_features = tf.matmul(h1, W_fc1) + b_fc1
            texts_features = tf.nn.relu(texts_features)

        # Stack image and text features
        stacked_features = tf.stack([images_features, texts_features], axis=1)
        stacked_features = tf.reshape(stacked_features, [-1, 2 * fc1_size])

        W_softmax = tf.get_variable('W_softmax', [2 * fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(stacked_features, W_softmax) + b_softmax

        # Accuracy metrics
        accuracy = slim.metrics.streaming_accuracy(tf.cast(labels, tf.int32),
                                                   tf.cast(tf.argmax(logits, 1), tf.int32))

        # Choose the metrics to compute:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': accuracy,
        })

        for metric_name, metric_value in names_to_values.iteritems():
            tf.summary.scalar(metric_name, metric_value)

        log_dir = os.path.join(log_dir, mode)

        # Evaluate every eval_interval_secs secs or if not specified,
        # every time the checkpoint_dir changes
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            log_dir,
            num_evals=num_evals,
            eval_op=names_to_updates.values())