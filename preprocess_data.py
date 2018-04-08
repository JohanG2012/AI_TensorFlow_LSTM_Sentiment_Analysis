import numpy as np
import pandas as pd
import utils as utl
from collections import Counter


def preprocess_data():
    # GET DATA
    data = pd.read_csv("data/StockTwits_SPY_Sentiment_2017.gz",
                       encoding="utf-8",
                       compression="gzip",
                       index_col=0)

    # GET MESSAGES AND VALUS
    messages = data.message.values
    labels = data.sentiment.values

    messages = np.array([utl.preprocess_ST_message(message) for message in messages])

    full_lexicon = " ".join(messages).split()
    vocab_to_int, int_to_vocab = utl.create_lookup_tables(full_lexicon)

    messages_lens = Counter([len(x) for x in messages])
    print("Zero-length messages: {}".format(messages_lens[0]))
    print("Maximum message length: {}".format(max(messages_lens)))
    print("Average message length: {}".format(np.mean([len(x) for x in messages])))

    messages, labels = utl.drop_empty_messages(messages, labels)

    messages = utl.encode_ST_messages(messages, vocab_to_int)
    labels = utl.encode_ST_labels(labels)

    messages = utl.zero_pad_messages(messages, seq_len=244)

    train_x, val_x, test_x, train_y, val_y, test_y = utl.train_val_test_split(messages, labels, split_frac=0.80)
    return train_x, val_x, test_x, train_y, val_y, test_y, vocab_to_int
