import tensorflow as tf
import numpy as np

from time import time
from sklearn.model_selection import train_test_split
from text_model.text_preprocessing import preprocess_df

_RANDOM_SEED = 0

class CharModel():
    def __init__(self, config):
        self.config = config
        vocab_size = config['vocab_size']
        embedding_dim = config['embedding_dim']
        post_size = config['post_size']
        fc1_size = config['fc1_size']
        nb_emotions = config['nb_emotions']
        dropout = config['dropout']
        max_grad_norm = config['max_grad_norm']
        initial_lr = config['initial_lr']
        
        self.input_data = tf.placeholder(tf.int32, [None, post_size])
        self.target = tf.placeholder(tf.int32, [None])
        self.learning_rate = tf.Variable(initial_lr, trainable=False)
        # Use a placeholder to turn off dropout during testing 
        self.keep_prob = tf.placeholder(tf.float32)
        # Placeholder for embedding weights
        self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        
        # Word embedding
        W_embedding = tf.get_variable('W_embedding', [vocab_size, embedding_dim], trainable=False)
        self.embedding_init = W_embedding.assign(self.embedding_placeholder)
        input_embed = tf.nn.embedding_lookup(W_embedding, self.input_data)
        input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)

        # Rescale the mean by the actual number of non-zero values.
        nb_finite = tf.reduce_sum(tf.cast(tf.not_equal(input_embed_dropout, 0.0), tf.float32), axis=1)
        # If a post has zero finite elements, replace nb_finite by 1
        nb_finite = tf.where(tf.equal(nb_finite, 0.0), tf.ones_like(nb_finite), nb_finite)
        self.h1 = tf.reduce_mean(input_embed_dropout, axis=1) * post_size / nb_finite

        # Fully connected layer
        W_fc1 = tf.get_variable('W_fc1', [embedding_dim, fc1_size])
        b_fc1 = tf.get_variable('b_fc1', [fc1_size])
        h2 = tf.matmul(self.h1, W_fc1) + b_fc1
        h2 = tf.nn.relu(h2)

        W_softmax = tf.get_variable('W_softmax', [fc1_size, nb_emotions])
        b_softmax = tf.get_variable('b_softmax', [nb_emotions])
        logits = tf.matmul(h2, W_softmax) + b_softmax
        labels = tf.one_hot(self.target, nb_emotions)
        # Cross-entropy loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        # Use gradient cliping
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.apply_gradients(zip(grads, trainable_vars),
                                                    global_step=tf.contrib.framework.get_or_create_global_step())
        #self.sample = tf.multinomial(tf.reshape(logits, [-1, vocab_size]), 1)
        correct_pred = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.target)
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def _shuffling(X, y):
    p = np.random.permutation(X.shape[0])
    return X[p], y[p]

def run_model(sess, model, X, y, is_training, model_gen=None):
    batch_size = model.config['batch_size']
    dropout = model.config['dropout']
    initial_lr = model.config['initial_lr']
    lr_decay = model.config['lr_decay']
    max_epoch_no_decay = model.config['max_epoch_no_decay']
    nb_epochs = model.config['nb_epochs']
    
    nb_batches = X.shape[0] / batch_size
    if is_training:
        # Iteration to print at
        print_iter = list(np.linspace(0, nb_batches - 1, 11).astype(int))
        dropout_param = dropout
        ops = [model.loss, model.accuracy, model.train_step]
    else:
        dropout_param = 1.0
        ops = [model.loss, model.accuracy, tf.no_op()]

    for e in range(nb_epochs):
        print ('Epoch: {0}'.format(e + 1))
        lr_decay = lr_decay ** max(e + 1 - max_epoch_no_decay, 0)
        sess.run(tf.assign(model.learning_rate, initial_lr * lr_decay))

        total_loss = 0.0
        total_accuracy = 0.0
        nb_iter = 0.0
        loss_history = []
        t0 = time()
        X, y = _shuffling(X, y)
        X_reshaped = X[: (nb_batches * batch_size), :].reshape((nb_batches, batch_size, -1))
        y_reshaped = y[: (nb_batches * batch_size)].reshape((nb_batches, batch_size))
        for i in range(nb_batches):
            curr_input = X_reshaped[i, :, :]
            curr_target = y_reshaped[i, :]
            curr_loss, curr_acc, _ = sess.run(ops, feed_dict={model.input_data: curr_input, 
                                                              model.target: curr_target,
                                                              model.keep_prob: dropout_param})
            total_loss += curr_loss
            total_accuracy += curr_acc
            nb_iter += 1
            loss_history.append(curr_loss)

            if (is_training and i in print_iter):
                print('{0:.0f}%  loss = {1:.3f}, accuracy = {2:.3f}, speed = {3:.0f} pps'\
                      .format(print_iter.index(i) * 10, 
                              total_loss / nb_iter, total_accuracy / nb_iter,
                              (nb_iter * batch_size) / (time() - t0)))
                
        if is_training:
            pass
            #first_char = np.array([[4]])
            #samples = generate_chars(sess, model_gen, first_char, 2000)
            #generated_chars = map(lambda x: model_gen.config['id_to_char'][x], samples)
            #np.save('generated_chars.npy', np.array(generated_chars))
            #generated_chars = np.load('generated_chars.npy')
            #print('Generated characters:')
            # Need to add encode('utf-8') because when using the server,
            # sys.stdout.encoding is None
            #print(u''.join(list(generated_chars)).replace(u'_', u' ').encode('utf-8'))
        else:
            print('Loss = {0:.3f}, accuracy = {1:.3f}, speed = {2:.0f} pps'\
                  .format(total_loss / nb_iter, total_accuracy / nb_iter,
                          (nb_iter * batch_size) / (time() - t0)))

        #if (is_training and show_loss_graph):
            #plt.plot(perplexity_history)
            #plt.grid(True)
            #plt.title('Epoch {0}'.format(e + 1))
            #plt.xlabel('Mini-batch number')
            #plt.ylabel('Perplexity per mini-batch')
            #plt.show()
            
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
            m_train = CharModel(config)
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

def main_text():
    text_dir = 'text_model'
    emb_dir = 'embedding_weights'
    filename = 'glove.6B.50d.txt'
    emb_name = 'glove'
    emotions = ['happy', 'sad', 'angry', 'scared', 'disgusted', 'surprised']
    post_size = 200
    df_all, word_to_id, embedding = preprocess_df(text_dir, emb_dir, filename, emb_name, emotions, post_size)

    X = np.stack(df_all['text_list'])
    y = df_all['search_query'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=_RANDOM_SEED)

    id_to_word = {i: k for k, i in word_to_id.iteritems()}
    config = {'word_to_id': word_to_id,
              'id_to_word': id_to_word,
              'batch_size': 128,
              'vocab_size': len(word_to_id),
              'embedding_dim': embedding.shape[1],
              'post_size': post_size,
              'fc1_size': 2048,
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
        print('Training:')
        init_scale = config['init_scale']
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)    
        with tf.variable_scope('Model', reuse=None, initializer=initializer):
            config['nb_epochs'] = 10
            m_train = CharModel(config)
        sess.run(tf.global_variables_initializer())
        sess.run(m_train.embedding_init, feed_dict={m_train.embedding_placeholder: embedding})
        # Characters generation
        #with tf.variable_scope('Model', reuse=True):
            #config_gen = dict(config)
            #config_gen['batch_size'] = 1
            #config_gen['num_steps'] = 1
            #m_gen = CharModel(config_gen)
        run_model(sess, m_train, X_train, y_train, is_training=True)
        
        print('\nValidation:')
        with tf.variable_scope('Model', reuse=True):
            config['nb_epochs'] = 1
            m_valid = CharModel(config)
        run_model(sess, m_valid, X_valid, y_valid, is_training=False)
        
        #print('\nTest:')
        #with tf.variable_scope('Model', reuse=True):
         #   m_test =  CharModel(config)
        #run_model(sess, m_test, test_data, is_training=False)
        print('Finished')