import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.learning import train_step
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from scipy.ndimage.filters import gaussian_filter1d

from slim.preprocessing import inception_preprocessing
from image_model import inception_v1
from datasets import dataset_utils
from text_model.text_preprocessing import _load_embedding_weights_glove
from image_model.im_model import load_batch_with_text, get_init_fn
from datasets.convert_to_dataset import get_split_with_text
import matplotlib.pyplot as plt

_POST_SIZE = 200

def train_deep_sentiment(dataset_dir, checkpoints_dir, train_dir, num_steps, initial_lr):
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

        learning_rate = tf.Variable(initial_lr, trainable=False)
        lr_rate_placeholder = tf.placeholder(tf.float32)
        lr_rate_assign = learning_rate.assign(lr_rate_placeholder)

        dataset = get_split_with_text('train', dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, labels = load_batch_with_text(dataset, height=image_size, width=image_size)
        
        im_features_size = 128
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, num_classes=im_features_size, is_training=True)

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

            fc1_size = 2048
            # Fully connected layer
            W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
            b_fc1 = tf.get_variable('b_fc1', [fc1_size])
            texts_features = tf.matmul(h1, W_fc1) + b_fc1
            texts_features = tf.nn.relu(texts_features)

        # Concatenate image and text features
        concat_features = tf.concat([images_features, texts_features], axis=1)

        # Fully connected layer

        W_softmax = tf.get_variable('W_softmax', [im_features_size + fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(concat_features, W_softmax) + b_softmax
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

        nb_batches = dataset.num_samples / 32
        def train_step_fn(session, *args, **kwargs):
            # Decaying learning rate every epoch
            if train_step_fn.step % (nb_batches) == 0:
                lr_decay = 0.5 ** train_step_fn.epoch
                session.run(lr_rate_assign, feed_dict={lr_rate_placeholder: initial_lr * lr_decay})
                print('New learning rate: {0}'. format(initial_lr * lr_decay))
                train_step_fn.epoch += 1

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
        train_step_fn.epoch = 0

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
        
        im_features_size = 128
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, num_classes=im_features_size, is_training=True)

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

            fc1_size = 2048
            # Fully connected layer
            W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
            b_fc1 = tf.get_variable('b_fc1', [fc1_size])
            texts_features = tf.matmul(h1, W_fc1) + b_fc1
            texts_features = tf.nn.relu(texts_features)

        # Concatenate image and text features
        concat_features = tf.concat([images_features, texts_features], axis=1)

        W_softmax = tf.get_variable('W_softmax', [im_features_size + fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(concat_features, W_softmax) + b_softmax

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
        # tf.get_variable variables are also restored
        slim.evaluation.evaluation_loop(
            '',
            checkpoint_dir,
            log_dir,
            num_evals=num_evals,
            eval_op=names_to_updates.values())

def deprocess_image(np_image):
    return (np_image - 0.5) / 2.0

def blur_image(np_image, sigma=1):
    np_image = gaussian_filter1d(np_image, sigma, axis=1)
    np_image = gaussian_filter1d(np_image, sigma, axis=2)
    return np_image

def class_visualisation(label, learning_rate, checkpoint_dir):
    """Visualise class with gradient ascent.
    
    Parameters:
        label: Label to visualise.
        learning_rate: Learning rate of the gradient ascent.
        checkpoint_dir: Checkpoint of the saved model during training.
    """
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        image_size = inception_v1.default_image_size
        image = tf.placeholder(tf.float32, [1, image_size, image_size, 3])

        # Text model
        text_dir = 'text_model'
        emb_dir = 'embedding_weights'
        filename = 'glove.6B.50d.txt'
        vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
        vocab_size, embedding_dim = embedding.shape
        word_to_id = dict(zip(vocabulary, range(vocab_size)))

        # Create text with only unknown words
        text = tf.constant(np.ones((1, _POST_SIZE), dtype=np.int32) * vocab_size)

        im_features_size = 128
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(image, num_classes=im_features_size, is_training=True)

        # Unknown words = vector with zeros
        embedding = np.concatenate([embedding, np.zeros((1, embedding_dim))])
        word_to_id['<ukn>'] = vocab_size

        vocab_size = len(word_to_id)
        nb_emotions = 6
        with tf.variable_scope('Text'):
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        
            # Word embedding
            W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
            embedding_init = W_embedding.assign(embedding_placeholder)
            input_embed = tf.nn.embedding_lookup(W_embedding, text)
            #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

            # Rescale the mean by the actual number of non-zero values.
            nb_finite = tf.reduce_sum(tf.cast(tf.not_equal(input_embed, 0.0), tf.float32), axis=1)
            # If a post has zero finite elements, replace nb_finite by 1
            nb_finite = tf.where(tf.equal(nb_finite, 0.0), tf.ones_like(nb_finite), nb_finite)
            h1 = tf.reduce_mean(input_embed, axis=1) * _POST_SIZE / nb_finite

            fc1_size = 2048
            # Fully connected layer
            W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
            b_fc1 = tf.get_variable('b_fc1', [fc1_size])
            texts_features = tf.matmul(h1, W_fc1) + b_fc1
            texts_features = tf.nn.relu(texts_features)

        # Concatenate image and text features
        concat_features = tf.concat([images_features, texts_features], axis=1)

        W_softmax = tf.get_variable('W_softmax', [im_features_size + fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(concat_features, W_softmax) + b_softmax

        class_score = logits[:, label]
        l2_reg = 0.001
        regularisation = l2_reg * tf.square(tf.norm(image))
        obj_function = class_score  - regularisation
        grad_obj_function = tf.gradients(obj_function, image)[0]
        grad_normalized = grad_obj_function / tf.norm(grad_obj_function)

        # Initialise image
        image_init = tf.random_normal([image_size, image_size, 3])
        image_init = inception_preprocessing.preprocess_image(image_init, image_size, image_size, is_training=False)
        image_init = tf.expand_dims(image_init, 0)

        # Load model
        checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
        scaffold = monitored_session.Scaffold(
            init_op=None, init_feed_dict=None,
            init_fn=None, saver=None)
        session_creator = monitored_session.ChiefSessionCreator(
            scaffold=scaffold,
            checkpoint_filename_with_path=checkpoint_path,
            master='',
            config=None)

        blur_every = 10
        max_jitter = 16
        show_every = 50
        clip_percentile = 20

        with monitored_session.MonitoredSession(
            session_creator=session_creator, hooks=None) as session:
            np_image = session.run(image_init)
            num_iterations = 500
            for i in range(num_iterations):
                # Randomly jitter the image a bit
                ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)
                np_image = np.roll(np.roll(np_image, ox, 1), oy, 2)

                # Update image
                grad_update = session.run(grad_normalized, feed_dict={image: np_image})
                np_image += learning_rate * grad_update

                # Undo the jitter
                np_image = np.roll(np.roll(np_image, -ox, 1), -oy, 2)

                # As a regularizer, clip and periodically blur
                #np_image = np.clip(np_image, -0.2, 0.8)
                # Set pixels with small norm to zero
                min_norm = np.percentile(np_image, clip_percentile)
                np_image[np_image < min_norm] = 0.0
                if i % blur_every == 0:
                    np_image = blur_image(np_image, sigma=0.5)

                if i % show_every == 0 or i == (num_iterations - 1):
                    plt.imshow(deprocess_image(np_image[0]))
                    plt.title('Iteration %d / %d' % (i + 1, num_iterations))
                    plt.gcf().set_size_inches(4, 4)
                    plt.axis('off')
                    plt.show()


def word_most_relevant(label, checkpoint_dir):
    """Compute gradient of W_embedding to get the word most relevant to a label.
    
    Parameters:
        label: Label to get the words most relevant to.
        checkpoint_dir: Checkpoint of the saved model during training.
    """
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        image_size = inception_v1.default_image_size
        image = tf.placeholder(tf.float32, [1, image_size, image_size, 3])

        # Text model
        text_dir = 'text_model'
        emb_dir = 'embedding_weights'
        filename = 'glove.6B.50d.txt'
        vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
        vocab_size, embedding_dim = embedding.shape
        word_to_id = dict(zip(vocabulary, range(vocab_size)))

        text = tf.placeholder(tf.int32, [1, _POST_SIZE])

        im_features_size = 128
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(image, num_classes=im_features_size, is_training=True)

        # Unknown words = vector with zeros
        embedding = np.concatenate([embedding, np.zeros((1, embedding_dim))])
        word_to_id['<ukn>'] = vocab_size

        vocab_size = len(word_to_id)
        nb_emotions = 6
        with tf.variable_scope('Text'):
            embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        
            # Word embedding
            W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
            embedding_init = W_embedding.assign(embedding_placeholder)
            input_embed = tf.nn.embedding_lookup(W_embedding, text)
            #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

            # Rescale the mean by the actual number of non-zero values.
            nb_finite = tf.reduce_sum(tf.cast(tf.not_equal(input_embed, 0.0), tf.float32), axis=1)
            # If a post has zero finite elements, replace nb_finite by 1
            nb_finite = tf.where(tf.equal(nb_finite, 0.0), tf.ones_like(nb_finite), nb_finite)
            h1 = tf.reduce_mean(input_embed, axis=1) * _POST_SIZE / nb_finite

            fc1_size = 2048
            # Fully connected layer
            W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
            b_fc1 = tf.get_variable('b_fc1', [fc1_size])
            texts_features = tf.matmul(h1, W_fc1) + b_fc1
            texts_features = tf.nn.relu(texts_features)

        # Concatenate image and text features
        concat_features = tf.concat([images_features, texts_features], axis=1)

        W_softmax = tf.get_variable('W_softmax', [im_features_size + fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(concat_features, W_softmax) + b_softmax

        class_score = logits[:, label]
        #l2_reg = 0.001
        #regularisation = l2_reg * tf.square(tf.norm(image))
        obj_function = class_score  #- regularisation
        grad_obj_function = tf.gradients(obj_function, W_embedding)[0]
        grad_norm = tf.norm(grad_obj_function)
        #grad_normalized = grad_obj_function / tf.norm(grad_obj_function)

        # Initialise image
        image_init = tf.random_normal([image_size, image_size, 3])
        image_init = inception_preprocessing.preprocess_image(image_init, image_size, image_size, is_training=False)
        image_init = tf.expand_dims(image_init, 0)

        # Load model
        checkpoint_path = tf_saver.latest_checkpoint(checkpoint_dir)
        scaffold = monitored_session.Scaffold(
            init_op=None, init_feed_dict=None,
            init_fn=None, saver=None)
        session_creator = monitored_session.ChiefSessionCreator(
            scaffold=scaffold,
            checkpoint_filename_with_path=checkpoint_path,
            master='',
            config=None)

        with monitored_session.MonitoredSession(
            session_creator=session_creator, hooks=None) as session:
            np_image = session.run(image_init)
            grads = []
            
            for i in range(vocab_size):
                np_text = np.ones((1, _POST_SIZE), dtype=np.int32) * i
                grads.append(session.run(grad_norm, feed_dict={image: np_image, text: np_text}))
            return np.array(grads), vocabulary, word_to_id


def generate_text(checkpoints_dir, num_steps, initial_lr):
    """Generate text given an image.

    Parameters:
        dataset_dir: The directory containing the data.
        checkpoints_dir: The directory contained the pre-trained model.
        num_steps: The number of steps training the model.
        initial_lr: Initial learning rate.
    """
    dataset_dir = 'data'
    with tf.Graph().as_default():
        tf.logging.set_verbosity(tf.logging.INFO)

        learning_rate = tf.Variable(initial_lr, trainable=False)
        lr_rate_placeholder = tf.placeholder(tf.float32)
        lr_rate_assign = learning_rate.assign(lr_rate_placeholder)

        dataset = get_split_with_text('train', dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, labels = load_batch_with_text(dataset, height=image_size, width=image_size)
        
        im_features_size = 128
        # Create the model, use the default arg scope to configure the batch norm parameters.
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, num_classes=im_features_size, is_training=True)

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

            fc1_size = 2048
            # Fully connected layer
            W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
            b_fc1 = tf.get_variable('b_fc1', [fc1_size])
            texts_features = tf.matmul(h1, W_fc1) + b_fc1
            texts_features = tf.nn.relu(texts_features)

        # Concatenate image and text features
        concat_features = tf.concat([images_features, texts_features], axis=1)

        W_softmax = tf.get_variable('W_softmax', [im_features_size + fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(concat_features, W_softmax) + b_softmax
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

        nb_batches = dataset.num_samples / 32
        def train_step_fn(session, *args, **kwargs):
            # Decaying learning rate every epoch
            if train_step_fn.step % (nb_batches) == 0:
                lr_decay = 0.5 ** train_step_fn.epoch
                session.run(lr_rate_assign, feed_dict={lr_rate_placeholder: initial_lr * lr_decay})
                print('New learning rate: {0}'. format(initial_lr * lr_decay))
                train_step_fn.epoch += 1

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
        train_step_fn.epoch = 0

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



        