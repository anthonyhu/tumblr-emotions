import os
import tensorflow as tf
import numpy as np

from time import time
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.learning import train_step

from text_model.text_preprocessing import preprocess_df
from image_model import inception_v1
from text_model.text_preprocessing import _load_embedding_weights_glove
from image_model.im_model import load_batch_with_text
from datasets.convert_to_dataset import get_split_with_text

_RANDOM_SEED = 0
_CONFIG = {'mode': 'train',
           'dataset_dir': 'data',
           'text_dir': 'text_model',
           'emb_dir': 'embedding_weights',
           'filename': 'glove.6B.50d.txt',
           'initial_lr': 1e-3,
           'decay_factor': 0.3,
           'batch_size': 64,
           'rnn_size': 1024}


def _shuffling(X, y):
    p = np.random.permutation(X.shape[0])
    return X[p], y[p]


def _shuffling_rnn(X, seq_len, y):
    p = np.random.permutation(X.shape[0])
    return X[p], seq_len[p], y[p]


class TextModel():
    def __init__(self, config):
        self.config = config
        mode = config['mode']
        dataset_dir = config['dataset_dir']
        text_dir = config['text_dir']
        emb_dir = config['emb_dir']
        filename = config['filename']
        initial_lr = config['initial_lr']
        batch_size = config['batch_size']
        rnn_size = config['rnn_size']

        tf.logging.set_verbosity(tf.logging.INFO)

        self.learning_rate = tf.Variable(initial_lr, trainable=False)
        self.lr_rate_placeholder = tf.placeholder(tf.float32)
        self.lr_rate_assign = self.learning_rate.assign(self.lr_rate_placeholder)

        self.dataset = get_split_with_text(mode, dataset_dir)
        image_size = inception_v1.default_image_size
        images, _, texts, seq_lens, self.labels, _, _ = load_batch_with_text(self.dataset, batch_size, height=image_size, 
                                                                             width=image_size)

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

        W_softmax = tf.get_variable('W_softmax', [rnn_size, self.nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [self.nb_emotions])
        self.logits = tf.matmul(texts_features, W_softmax) + b_softmax


def train_text_model(train_dir, num_steps):
    """Train rnn text model.

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
        model = TextModel(_CONFIG)
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

            train_step_fn.step += 1
            return [total_loss, should_stop]
        
        train_step_fn.step = 0
        train_step_fn.epoch = 0

        # Run the training:
        final_loss = slim.learning.train(
            train_op,
            logdir=train_dir,
            save_interval_secs=600,
            save_summaries_secs=600,
            train_step_fn=train_step_fn,
            number_of_steps=num_steps)
            
    print('Finished training. Last batch loss {0:.3f}'.format(final_loss))

def evaluate_text_model(checkpoint_dir, log_dir, mode, num_evals):
    """Visualise results with: tensorboard --logdir=logdir. Now has train/validation curves on the same plot
    
    Parameters:
        checkpoint_dir: Checkpoint of the saved model during training.
        log_dir: Directory to save logs.
        mode: train or validation.
        num_evals: Number of batches to evaluate (mean of the batches is displayed).
    """
    with tf.Graph().as_default():
        _CONFIG['mode'] = mode
        model = TextModel(_CONFIG)

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


def generate_chars(sess, model, first_char, max_iteration):
    ops = [model.final_state, model.sample]
    current_char = first_char.copy()
    numpy_state = sess.run(model.initial_state)
    samples = []
    for i in range(max_iteration):
        # Sample from the multinomial distribution of the next character
        numpy_state, sample = sess.run(ops, feed_dict={model.input_data: current_char,
                                                       model.initial_state: numpy_state,
                                                       model.keep_prob: 1.0})
        samples.append(sample[0][0])
        current_char = sample
    return samples


def compute_sklearn_features():
    """Compute mean word embedding features for sklearn models.
    """
    text_dir = 'text_model'
    emb_dir = 'embedding_weights'
    filename = 'glove.6B.50d.txt'
    emb_name = 'glove'
    emotions = ['happy', 'sad', 'angry', 'scared', 'disgusted', 'surprised']
    post_size = 200
    df_all, word_to_id, embedding = preprocess_df(text_dir, emb_dir, filename, emb_name, emotions, post_size)

    X = np.stack(df_all['text_list'])
    y = df_all['search_query'].values

    id_to_word = {i: k for k, i in word_to_id.iteritems()}
    config = {'word_to_id': word_to_id,
              'id_to_word': id_to_word,
              'batch_size': 128,
              'vocab_size': len(word_to_id),
              'embedding_dim': embedding.shape[1],
              'post_size': post_size,
              'fc1_size': 16,
              'nb_emotions': len(emotions),
              'dropout': 1.0, # Proba to keep neurons
              'max_grad_norm': 5.0, # Maximum norm of gradient
              'init_scale': 0.1, # Weights initialization scale
              'initial_lr': 1e-3,
              'lr_decay': 0.5,
              'max_epoch_no_decay': 2, # Number of epochs without decaying learning rate
              'nb_epochs': 10} # Maximum number of epochs
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        print('Computing sklearn features:')
        init_scale = config['init_scale']
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)    
        with tf.variable_scope('Model', reuse=None, initializer=initializer):
            config['nb_epochs'] = 1
            m_train = WordModel(config)
        sess.run(tf.global_variables_initializer())
        sess.run(m_train.embedding_init, feed_dict={m_train.embedding_placeholder: embedding})

        batch_size = m_train.config['batch_size']
        initial_lr = m_train.config['initial_lr']
        
        nb_batches = X.shape[0] / batch_size
        dropout_param = 1.0
        ops = m_train.h1
        
        sess.run(tf.assign(m_train.learning_rate, initial_lr))

        X, y = _shuffling(X, y)
        X_reshaped = X[: (nb_batches * batch_size), :].reshape((nb_batches, batch_size, -1))
        y_reshaped = y[: (nb_batches * batch_size)].reshape((nb_batches, batch_size))
        h1_list = []
        for i in range(nb_batches):
            curr_input = X_reshaped[i, :, :]
            curr_target = y_reshaped[i, :]
            h1_features = sess.run(ops, feed_dict={m_train.input_data: curr_input, 
                                                   m_train.target: curr_target,
                                                   m_train.keep_prob: dropout_param})
            h1_list.append(h1_features)

        X_sklearn = np.vstack(h1_list)
        y_sklearn = y_reshaped.reshape((-1))
        print('Finished')
        return X_sklearn, y_sklearn