import tensorflow as tf
import preprocess_data as pd
import utils as utl
import numpy as np
import train_lstm as tl

train_x, val_x, test_x, train_y, val_y, test_y, vocab_to_int = pd.preprocess_data()
batch_size = 256
vocab_size = len(vocab_to_int) + 1 #add one for padding
embed_size = 300
lstm_sizes = [128, 64]
epochs = 50
learning_rate = 0.1
keep_prob = 0.5

def test_network(model_dir, batch_size, test_x, test_y):

    inputs_, labels_, keep_prob_ = tl.model_inputs()
    embed = tl.build_embedding_layer(inputs_, vocab_size, embed_size)
    initial_state, lstm_outputs, lstm_cell, final_state = tl.build_lstm_layers(lstm_sizes, embed, keep_prob_, batch_size)
    predictions, loss, optimizer = tl.build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = tl.build_accuracy(predictions, labels_)

    saver = tf.train.Saver()

    test_acc = []
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_dir))
        test_state = sess.run(lstm_cell.zero_state(batch_size, tf.float32))
        for ii, (x, y) in enumerate(utl.get_batches(test_x, test_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob_: 1,
                    initial_state: test_state}
            batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
            test_acc.append(batch_acc)
        print("Test Accuracy: {:.3f}".format(np.mean(test_acc)))

with tf.Graph().as_default():
    test_network('checkpoints', batch_size, test_x, test_y)
