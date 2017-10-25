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

_POST_SIZE = 50
_CONFIG = {'mode': 'train',
           'dataset_dir': 'data',
           'text_dir': 'text_model',
           'emb_dir': 'embedding_weights',
           'filename': 'glove.6B.50d.txt',
           'initial_lr': 1e-3,
           'decay_factor': 0.3,
           'batch_size': 64,
           'im_features_size': 256,
           'rnn_size': 1024,
           'final_endpoint': 'Mixed_5c',
           'fc_size': 512}

class DeepSentiment():
    def __init__(self, config):
        self.config = config
        mode = config['mode']
        dataset_dir = config['dataset_dir']
        text_dir = config['text_dir']
        emb_dir = config['emb_dir']
        filename = config['filename']
        initial_lr = config['initial_lr']
        batch_size = config['batch_size']
        im_features_size = config['im_features_size']
        rnn_size = config['rnn_size']
        final_endpoint = config['final_endpoint']

        tf.logging.set_verbosity(tf.logging.INFO)

        self.learning_rate = tf.Variable(initial_lr, trainable=False)
        self.lr_rate_placeholder = tf.placeholder(tf.float32)
        self.lr_rate_assign = self.learning_rate.assign(self.lr_rate_placeholder)

        self.dataset = get_split_with_text(mode, dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, seq_lens, self.labels, self.post_ids, self.days = load_batch_with_text(self.dataset, batch_size, 
            height=image_size, width=image_size)
            
        # Create the model, use the default arg scope to configure the batch norm parameters.
        is_training = (mode == 'train')
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, final_endpoint=final_endpoint,
                num_classes=im_features_size, is_training=is_training)

        # Text model
        vocabulary, self.embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
        vocab_size, embedding_dim = self.embedding.shape
        word_to_id = dict(zip(vocabulary, range(vocab_size)))
        # Unknown words = vector with zeros
        self.embedding = np.concatenate([self.embedding, np.zeros((1, embedding_dim))])
        word_to_id['<ukn>'] = vocab_size

        vocab_size = len(word_to_id)
        self.nb_emotions = self.dataset.num_classes
        with tf.variable_scope('Text'):
            # Word embedding
            W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding_init = W_embedding.assign(self.embedding_placeholder)
            input_embed = tf.nn.embedding_lookup(W_embedding, texts)
            #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

            # LSTM
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, input_embed, sequence_length=seq_lens, dtype=tf.float32)
            # Need to convert seq_lens to int32 for stack
            texts_features = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), tf.cast(seq_lens, tf.int32) - 1], axis=1))

        # Concatenate image and text features
        self.concat_features = tf.concat([images_features, texts_features], axis=1)

        W_softmax = tf.get_variable('W_softmax', [im_features_size + rnn_size, self.nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [self.nb_emotions])
        self.logits = tf.matmul(self.concat_features, W_softmax) + b_softmax

class DeepSentiment2():
    def __init__(self, config):
        self.config = config
        mode = config['mode']
        dataset_dir = config['dataset_dir']
        text_dir = config['text_dir']
        emb_dir = config['emb_dir']
        filename = config['filename']
        initial_lr = config['initial_lr']
        batch_size = config['batch_size']
        im_features_size = config['im_features_size']
        rnn_size = config['rnn_size']
        final_endpoint = config['final_endpoint']
        fc_size = config['fc_size']

        tf.logging.set_verbosity(tf.logging.INFO)

        self.learning_rate = tf.Variable(initial_lr, trainable=False)
        self.lr_rate_placeholder = tf.placeholder(tf.float32)
        self.lr_rate_assign = self.learning_rate.assign(self.lr_rate_placeholder)

        self.dataset = get_split_with_text(mode, dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, seq_lens, self.labels, self.post_ids, self.days = load_batch_with_text(self.dataset, batch_size, 
            height=image_size, width=image_size)
            
        # Create the model, use the default arg scope to configure the batch norm parameters.
        is_training = (mode == 'train')
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, final_endpoint=final_endpoint,
                num_classes=im_features_size, is_training=is_training)

        # Text model
        vocabulary, self.embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
        vocab_size, embedding_dim = self.embedding.shape
        word_to_id = dict(zip(vocabulary, range(vocab_size)))
        # Unknown words = vector with zeros
        self.embedding = np.concatenate([self.embedding, np.zeros((1, embedding_dim))])
        word_to_id['<ukn>'] = vocab_size

        vocab_size = len(word_to_id)
        self.nb_emotions = self.dataset.num_classes
        with tf.variable_scope('Text'):
            # Word embedding
            W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding_init = W_embedding.assign(self.embedding_placeholder)
            input_embed = tf.nn.embedding_lookup(W_embedding, texts)
            #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

            # LSTM
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, input_embed, sequence_length=seq_lens, dtype=tf.float32)
            # Need to convert seq_lens to int32 for stack
            texts_features = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), tf.cast(seq_lens, tf.int32) - 1], axis=1))

        # Concatenate image and text features
        self.concat_features = tf.concat([images_features, texts_features], axis=1)

        # Dense layer
        W_fc = tf.get_variable('W_fc', [im_features_size + rnn_size, fc_size])
        b_fc = tf.get_variable('b_fc', [fc_size])
        dense_layer = tf.matmul(self.concat_features, W_fc) + b_fc
        dense_layer_relu = tf.nn.relu(dense_layer)

        W_softmax = tf.get_variable('W_softmax', [fc_size, self.nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [self.nb_emotions])
        self.logits = tf.matmul(dense_layer_relu, W_softmax) + b_softmax

def train_deep_sentiment(checkpoints_dir, train_dir, num_steps):
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
        model = DeepSentiment2(_CONFIG)
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

            # Initialise embedding weights
            if train_step_fn.step == 0:
                session.run(model.embedding_init, feed_dict={model.embedding_placeholder: model.embedding})
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

def evaluate_deep_sentiment(checkpoint_dir, log_dir, mode, num_evals):
    """Visualise results with: tensorboard --logdir=logdir. Now has train/validation curves on the same plot
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
        log_dir: Directory to save logs.
        mode: train or validation.
        num_evals: Number of batches to evaluate (mean of the batches is displayed).
    """
    with tf.Graph().as_default():
        config = _CONFIG.copy()
        config['mode'] = mode
        model = DeepSentiment2(config)

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


def correlation_matrix(nb_batches, checkpoint_dir):
    """Computes logits and labels of the input posts and save them as numpy files.
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
    """
    with tf.Graph().as_default():
        config = _CONFIG.copy()
        config['mode'] = 'validation'
        model = DeepSentiment(config)

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

        posts_logits = []
        posts_labels = []
        with monitored_session.MonitoredSession( # Generate queue
            session_creator=session_creator, hooks=None) as session:
            for i in range(nb_batches):
                np_logits, np_labels = session.run([model.logits, model.labels])
                posts_logits.append(np_logits)
                posts_labels.append(np_labels)

    posts_logits, posts_labels = np.vstack(posts_logits), np.hstack(posts_labels)
    np.save('data/posts_logits.npy', posts_logits)
    np.save('data/posts_labels.npy', posts_labels)
    return posts_logits, posts_labels

def word_most_relevant(top_words, num_classes, checkpoint_dir):
    """Compute gradient of W_embedding to get the word most relevant to a label.
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
    """
    with tf.Graph().as_default():
        config = _CONFIG.copy()
        mode = 'validation'
        dataset_dir = config['dataset_dir']
        text_dir = config['text_dir']
        emb_dir = config['emb_dir']
        filename = config['filename']
        initial_lr = config['initial_lr']
        #batch_size = config['batch_size']
        im_features_size = config['im_features_size']
        rnn_size = config['rnn_size']
        final_endpoint = config['final_endpoint']

        tf.logging.set_verbosity(tf.logging.INFO)

        batch_size = 50
        image_size = inception_v1.default_image_size
        images = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
        texts = tf.placeholder(tf.int32, [batch_size, _POST_SIZE])
        seq_lens = tf.placeholder(tf.int32, [batch_size])

        #self.learning_rate = tf.Variable(initial_lr, trainable=False)
        #self.lr_rate_placeholder = tf.placeholder(tf.float32)
        #self.lr_rate_assign = self.learning_rate.assign(self.lr_rate_placeholder)

        #self.dataset = get_split_with_text(mode, dataset_dir)
        #image_size = inception_v1.default_image_size
        #images, _, texts, seq_lens, self.labels = load_batch_with_text(self.dataset, batch_size, height=image_size, width=image_size)
            
        # Create the model, use the default arg scope to configure the batch norm parameters.
        is_training = (mode == 'train')
        with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
            images_features, _ = inception_v1.inception_v1(images, final_endpoint=final_endpoint,
                num_classes=im_features_size, is_training=is_training)

        # Text model
        vocabulary, embedding = _load_embedding_weights_glove(text_dir, emb_dir, filename)
        vocab_size, embedding_dim = embedding.shape
        word_to_id = dict(zip(vocabulary, range(vocab_size)))
        # Unknown words = vector with zeros
        embedding = np.concatenate([embedding, np.zeros((1, embedding_dim))])
        word_to_id['<ukn>'] = vocab_size

        vocab_size = len(word_to_id)
        nb_emotions = num_classes
        with tf.variable_scope('Text'):
            # Word embedding
            W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
            #self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            #self.embedding_init = W_embedding.assign(self.embedding_placeholder)
            input_embed = tf.nn.embedding_lookup(W_embedding, texts)
            #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

            # LSTM
            cell = tf.contrib.rnn.BasicLSTMCell(rnn_size)
            rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, input_embed, sequence_length=seq_lens, dtype=tf.float32)
            # Need to convert seq_lens to int32 for stack
            texts_features = tf.gather_nd(rnn_outputs, tf.stack([tf.range(batch_size), tf.cast(seq_lens, tf.int32) - 1], axis=1))

        # Concatenate image and text features
        concat_features = tf.concat([images_features, texts_features], axis=1)

        W_softmax = tf.get_variable('W_softmax', [im_features_size + rnn_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(concat_features, W_softmax) + b_softmax
        
        # Initialise image
        #image_init = tf.random_normal([image_size, image_size, 3])
        #image_init = inception_preprocessing.preprocess_image(image_init, image_size, image_size, is_training=False)
        #image_init = tf.expand_dims(image_init, 0)

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

            nb_iter = len(top_words) / batch_size
            scores = []
            for i in range(nb_iter):
                np_images = np.zeros((batch_size, image_size, image_size, 3))
                np_texts = np.ones((batch_size, _POST_SIZE), dtype=np.int32) * (vocab_size - 1)
                np_texts[:, 0] = top_words[i*batch_size : (i+1)*batch_size]
                np_seq_lens = np.ones(batch_size, dtype=np.int32)
                scores.append(session.run(logits, feed_dict={images: np_images, texts: np_texts, seq_lens: np_seq_lens}))
    scores = np.vstack(scores)
    np.save('data/top_words_scores.npy', scores)
    np.save('data/top_words.npy', top_words)
    return scores, vocabulary, word_to_id


def outliers_detection(checkpoint_dir):
    """Find outliers using Euclidean distance in the last dense layer.
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
    """
    with tf.Graph().as_default():
        config = _CONFIG.copy()
        config['mode'] = 'validation'
        model = DeepSentiment(config)

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

        im_features_size = config['im_features_size']
        rnn_size = config['rnn_size']
        dense_mean = np.zeros((im_features_size + rnn_size))
        with monitored_session.MonitoredSession( # Generate queue
            session_creator=session_creator, hooks=None) as session:
            batch_size = config['batch_size']
            nb_batches = model.dataset.num_samples / batch_size
            for i in range(nb_batches):
                current_dense = session.run(model.concat_features)
                weight = float(i) * batch_size / ((i+1) * batch_size)
                dense_mean = weight * dense_mean + (1-weight) * current_dense.mean(axis=0)

            # Now look at outliers
            max_norms = np.zeros((batch_size))
            max_post_ids = np.zeros((batch_size))
            max_logits = np.zeros((batch_size, model.dataset.num_classes))
            for i in range(nb_batches):
                current_dense, np_post_ids, current_logits = session.run([model.concat_features, model.post_ids,
                    model.logits])
                current_diff = np.linalg.norm(current_dense - dense_mean, axis=1)
                for k in range(batch_size):
                    if current_diff[k] > max_norms[k]:
                        max_norms[k] = current_diff[k]
                        max_post_ids[k] = np_post_ids[k]
                        max_logits[k] = current_logits[k]
            
    np.save('data/max_norms.npy', max_norms)
    np.save('data/max_post_ids.npy', max_post_ids)
    np.save('data/max_logits.npy', max_logits)
    return max_norms, max_post_ids, max_logits

def day_of_week_trend(checkpoint_dir):
    """Compute day of week trend.
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
    """
    with tf.Graph().as_default():
        config = _CONFIG.copy()
        config['mode'] = 'validation'
        model = DeepSentiment(config)

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

        posts_logits = []
        posts_labels = []
        posts_days = []
        with monitored_session.MonitoredSession( # Generate queue
            session_creator=session_creator, hooks=None) as session:
            batch_size = config['batch_size']
            nb_batches = model.dataset.num_samples / batch_size
            for i in range(nb_batches):
                np_logits, np_labels, np_days = session.run([model.logits, model.labels, model.days])
                posts_logits.append(np_logits)
                posts_labels.append(np_labels)
                posts_days.append(np_days)

    posts_logits, posts_labels, posts_days = np.vstack(posts_logits), np.hstack(posts_labels), np.hstack(posts_days)
    np.save('data/posts_logits_week.npy', posts_logits)
    np.save('data/posts_labels_week.npy', posts_labels)
    np.save('data/posts_days_week.npy', posts_days)
    return posts_logits, posts_labels, posts_days
