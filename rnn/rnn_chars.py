import tensorflow as tf
import numpy as np
from time import time

class CharModel():
    def __init__(self, config):
        self.config = config
        batch_size = config['batch_size']
        num_steps = config['num_steps']
        vocab_size = config['vocab_size']
        hidden_size = config['hidden_size']
        num_layers = config['num_layers']
        dropout = config['dropout']
        max_grad_norm = config['max_grad_norm']
        initial_lr = config['initial_lr']
        
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.target = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.learning_rate = tf.Variable(initial_lr, trainable=False)
        # Use a placeholder to turn off dropout during testing 
        self.keep_prob = tf.placeholder(tf.float32)
        
        # Char embedding
        #embedding = tf.get_variable('embedding', [vocab_size, hidden_size])
        #input_embed = tf.nn.embedding_lookup(embedding, self.input_data)
        #input_embed_dropout = tf.nn.dropout(input_embed, self.keep_prob)
        input_data_one_hot = tf.one_hot(self.input_data, vocab_size)

        # LSTM
        def lstm_cell():
            return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0)
        def lstm_cell_dropout():
            return tf.contrib.rnn.DropoutWrapper(lstm_cell(), output_keep_prob=self.keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell_dropout() for _ in range(num_layers)], state_is_tuple=True)
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        state = self.initial_state
        outputs = []
        with tf.variable_scope('RNN'):
            for t in range(num_steps):
                if t > 0: tf.get_variable_scope().reuse_variables() # Reuse the weights in the LSTMs
                output, state = cell(input_data_one_hot[:, t, :], state)
                outputs.append(output)
        self.final_state = state

        h1 = tf.reshape(tf.stack(outputs, axis=1), [-1, hidden_size])
        W_softmax = tf.get_variable('W_softmax', [hidden_size, vocab_size])
        b_softmax = tf.get_variable('b_softmax', [vocab_size])
        logits = tf.matmul(h1, W_softmax) + b_softmax
        logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])
        # Use sequence loss for average over batch and sum across timesteps
        loss_vector = tf.contrib.seq2seq.sequence_loss(logits, self.target, weights=tf.ones([batch_size, num_steps]),
                                                       average_across_batch=True, average_across_timesteps=False)
        self.loss = tf.reduce_sum(loss_vector)
        # Use gradient cliping
        trainable_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, trainable_vars), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_step = optimizer.apply_gradients(zip(grads, trainable_vars),
                                                    global_step=tf.contrib.framework.get_or_create_global_step())
        self.sample = tf.multinomial(tf.reshape(logits, [-1, vocab_size]), 1)
        predict = tf.cast(tf.argmax(tf.reshape(logits, [-1, vocab_size]), 1), tf.int32)
        correct_pred = tf.equal(predict, tf.reshape(self.target, [-1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def run_model(sess, model, data, is_training, model_gen=None, show_loss_graph=False):
    batch_size = model.config['batch_size']
    num_steps = model.config['num_steps']
    dropout = model.config['dropout']
    initial_lr = model.config['initial_lr']
    lr_decay = model.config['lr_decay']
    max_epoch_no_decay = model.config['max_epoch_no_decay']
    nb_epochs = model.config['nb_epochs']
    
    batch_len = data.shape[0] / batch_size
    data = data[:batch_len * batch_size].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) / num_steps
    if is_training:
        # Iteration to print at
        print_iter = list(np.linspace(0, epoch_size - 1, 11).astype(int))
        dropout_param = dropout
        ops = [model.final_state, model.loss, model.accuracy, model.train_step]
    else:
        dropout_param = 1.0
        ops = [model.final_state, model.loss, model.accuracy, tf.no_op()]

    for e in range(nb_epochs):
        print ('Epoch: {0}'.format(e + 1))
        lr_decay = lr_decay ** max(e + 1 - max_epoch_no_decay, 0)
        sess.run(tf.assign(model.learning_rate, initial_lr * lr_decay))

        total_loss = 0.0
        total_accuracy = 0.0
        nb_iter = 0.0
        perplexity_history = []
        numpy_state = sess.run(model.initial_state)
        t0 = time()
        for i in range(epoch_size):
            curr_input = data[:, i * num_steps: (i + 1) * num_steps]
            # Target is the input shifted in time by 1
            curr_target = data[:, i * num_steps + 1: (i + 1) * num_steps + 1]
            numpy_state, curr_loss, curr_acc, _ = sess.run(ops,
                                                           feed_dict={model.input_data: curr_input, 
                                                                      model.target: curr_target,
                                                                      model.initial_state: numpy_state, 
                                                                      model.keep_prob: dropout_param})
            total_loss += curr_loss
            total_accuracy += curr_acc
            nb_iter += num_steps
            perplexity_history.append(np.exp(curr_loss / num_steps))

            if (is_training and i in print_iter):
                print('{0:.0f}% perplexity = {1:.3f}, accuracy = {2:.3f}, speed = {3:.0f} cps'\
                      .format(print_iter.index(i) * 10, 
                              np.exp(total_loss / nb_iter), total_accuracy / (i + 1),
                              (nb_iter * batch_size) / (time() - t0)))
                
        if is_training:
            first_char = np.array([[4]])
            samples = generate_chars(sess, model_gen, first_char, 2000)
            generated_chars = map(lambda x: model_gen.config['id_to_char'][x], samples)
            #np.save('generated_chars.npy', np.array(generated_chars))
            #generated_chars = np.load('generated_chars.npy')
            print('Generated characters:')
            # Need to add encode('utf-8') because when using the server,
            # sys.stdout.encoding is None
            print(u''.join(list(generated_chars)).replace(u'_', u' ').encode('utf-8'))
        else:
            print('Perplexity = {0:.3f}, accuracy = {1:.3f}, speed = {2:.0f} cps'\
                  .format(np.exp(total_loss / nb_iter), total_accuracy / (i + 1),
                          (nb_iter * batch_size) / (time() - t0)))

        if (is_training and show_loss_graph):
            plt.plot(perplexity_history)
            plt.grid(True)
            plt.title('Epoch {0}'.format(e + 1))
            plt.xlabel('Mini-batch number')
            plt.ylabel('Perplexity per mini-batch')
            plt.show()
            
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

def main():
    #train_data, valid_data, test_data, char_to_id = ptb_raw_data('data_chars', char=True)
    # Only contains 5M chars
    full_data = np.loadtxt('data_chars/happy_chars.txt', dtype=int)
    char_to_id = np.load('data_chars/char_to_id.npy').item()
    # Split into train/val/test 80/10/10
    train_split = (int)(full_data.shape[0] * 0.8)
    val_split = (int)(full_data.shape[0] * 0.9)
    train_data = full_data[:train_split]
    valid_data = full_data[train_split:val_split]
    test_data = full_data[val_split:]
    id_to_char = {i: k for k, i in char_to_id.iteritems()}
    config = {'char_to_id': char_to_id,
              'id_to_char': id_to_char,
              'batch_size': 20,
              'num_steps': 20,
              'vocab_size': len(char_to_id),
              'hidden_size': 300,
              'num_layers': 2, # Number of stacked LSTMs
              'dropout': 0.8, # Proba to keep neurons
              'max_grad_norm': 5.0, # Maximum norm of gradient
              'init_scale': 0.1, # Weights initialization scale
              'initial_lr': 1.0,
              'lr_decay': 0.5,
              'max_epoch_no_decay': 4, # Number of epochs without decaying learning rate
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
        # Characters generation
        with tf.variable_scope('Model', reuse=True):
            config_gen = dict(config)
            config_gen['batch_size'] = 1
            config_gen['num_steps'] = 1
            m_gen = CharModel(config_gen)
        run_model(sess, m_train, train_data, is_training=True, model_gen=m_gen)
        
        print('\nValidation:')
        with tf.variable_scope('Model', reuse=True):
            config['nb_epochs'] = 1
            m_valid = CharModel(config)
        run_model(sess, m_valid, valid_data, is_training=False)
        
        print('\nTest:')
        with tf.variable_scope('Model', reuse=True):
            m_test =  CharModel(config)
        run_model(sess, m_test, test_data, is_training=False)
        print('Finished')

if __name__ == '__main__':
    main()