import tensorflow as tf
import preprocess_data as pd
import utils as utl
import numpy as np

train_x, val_x, test_x, train_y, val_y, test_y, vocab_to_int = pd.preprocess_data()

# Model input pipes for data feed
def model_inputs():
    inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
    labels_ = tf.placeholder(tf.int32, [None, None], name="labels")
    keep_prob_ = tf.placeholder(tf.float32, name="keep_prob")
    return inputs_, labels_, keep_prob_

# Build, multi-dimensional vector of the current word.
def build_embedding_layer(inputs_, vocab_size, embed_size):
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1, 1))
    embed = tf.nn.embedding_lookup(embedding, inputs_)
    return embed

# Create LSTM Layer
def build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size):
    lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
    # Add dropout to the cell
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]
    # Stack up multiple LSTM layers, for deep learning
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)
    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state, sequence_length=keep_prob_)

    return initial_state, lstm_outputs, cell, final_state

# Build cost function and optimization
def build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate):
    predictions = tf.contrib.layers.fully_connected(lstm_outputs[:, -1], 1, activation_fn=tf.sigmoid)
    loss = tf.losses.mean_squared_error(labels_, predictions)
    optimzer = tf.train.AdadeltaOptimizer (learning_rate).minimize(loss)
    return predictions, loss, optimzer

def build_accuracy(predictions, labels_):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

# Build and train the network
def build_and_train_network(lstm_sizes, vocab_size, embed_size, epochs, batch_size, learning_rate, keep_prob, train_x, val_x, train_y, val_y):
    inputs_, labels_, keep_prob_ = model_inputs()
    embed = build_embedding_layer(inputs_, vocab_size, embed_size)
    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size)
    print(lstm_outputs)
    print(labels_)
    print(learning_rate)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = build_accuracy(predictions, labels_)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        n_batches = len(train_x)//batch_size
        for e in range(epochs):
            state = sess.run(initial_state)

            train_acc = []
            for ii, (x, y) in enumerate(utl.get_batches(train_x, train_y, batch_size), 1):
                feed = {inputs_: x,
                        labels_: y[:, None],
                        keep_prob_: keep_prob,
                        initial_state: state}
                loss_, state, _,  batch_acc = sess.run([loss, final_state, optimizer, accuracy], feed_dict=feed)
                train_acc.append(batch_acc)

                if (ii + 1) % n_batches == 0:

                    val_acc = []
                    val_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
                    for xx, yy in utl.get_batches(val_x, val_y, batch_size):
                        feed = {inputs_: xx,
                                labels_: yy[:, None],
                                keep_prob_: 1,
                                initial_state: val_state}
                        val_batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                        val_acc.append(val_batch_acc)

                    print("Epoch: {}/{}...".format(e+1, epochs),
                          "Batch: {}/{}...".format(ii+1, n_batches),
                          "Train Loss: {:.3f}...".format(loss_),
                          "Train Accruacy: {:.3f}...".format(np.mean(train_acc)),
                          "Val Accuracy: {:.3f}".format(np.mean(val_acc)))

        saver.save(sess, "checkpoints/sentiment.ckpt")

lstm_sizes = [128, 64]
vocab_size = len(vocab_to_int) + 1 #add one for padding
embed_size = 300
epochs = 50
batch_size = 256
learning_rate = 0.1
keep_prob = 0.5

with tf.Graph().as_default():
    build_and_train_network(lstm_sizes, vocab_size, embed_size, epochs, batch_size, learning_rate, keep_prob, train_x, val_x, train_y, val_y)
