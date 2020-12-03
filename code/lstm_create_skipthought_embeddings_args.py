from skip_thought import SkipThoughtEncoder
from data_loader import DataLoader
import numpy as np
np.random.seed(1)
import os
from time import time
import sys

# TO CREATE SKIPTHOUGHT EMBEDDINGS OF LSTM-GENERATED TRAINING DATA
# Takes 1 optional argument: number of samples to create the embedding for. If no argument, it encodes all samples.

lstm_generated_file_name = "../data/ours/generated_sentences_50epochs_POSTPROCESSED"
start_time = time()
LOAD_STORED_VAL_EMBEDDINGS = False
enc_lstm_data_filename = "../data/ours/enc_lstmfakeendings_data10_" + str(sys.argv[1]) + ".csv"
enc_lstm_context_data_filename = "../data/ours/enc_lstmfakeendings_data10_context_" + str(sys.argv[1]) + ".csv"

# load NEW TRAINING data with GENERATED FAKE ENDINGS
d = DataLoader()
lstm_right_ending_nr, lstm_context_sentences, lstm_ending_sentence1, lstm_ending_sentence2 = d.load_data_with_fake_endings(file_name=lstm_generated_file_name)
number_of_training_samples = len(lstm_right_ending_nr)
print("There are ", number_of_training_samples, " training samples.")

if not len(sys.argv) == 1:
    n_samples = int(sys.argv[1])
    lstm_right_ending_nr = lstm_right_ending_nr[:n_samples]
    lstm_context_sentences = lstm_context_sentences[:n_samples]
    lstm_ending_sentence1 = lstm_ending_sentence1[:n_samples]
    lstm_ending_sentence2 = lstm_ending_sentence2[:n_samples]

print("shape of trn context: ", lstm_context_sentences.shape)

if LOAD_STORED_VAL_EMBEDDINGS:
    print("Do nothing. ")

else:
    # initialize SkipThoughtEncoder
    s = SkipThoughtEncoder()
    s.init()

    # skipthought-encode lstm generated data
    lstm_enc_ending_sentence2 = np.array(list(map(s.encoder.encode, lstm_ending_sentence2)))
    lstm_enc_context_sentences = np.array(list(map(s.encoder.encode, lstm_context_sentences)))
    print("Dimension of lstm context vector: ", lstm_enc_context_sentences.shape)
    lstm_enc_ending_sentence1 = np.array(list(map(s.encoder.encode, lstm_ending_sentence1)))

    # stack encoded lstm data to write to file
    stacked_endings = np.concatenate((lstm_enc_ending_sentence1, lstm_enc_ending_sentence2), axis=1)
    print("shape of lstm_enc_context_sentences: ", lstm_enc_context_sentences.shape)
    print("shape of stacked_endings: ", stacked_endings.shape)
    stacked_lstm_enc_data = np.concatenate((lstm_enc_context_sentences, stacked_endings), axis=1)

    # expected data shape: number_of_lstm_samples x 6 x 4800
    print("Dimension of stacked lstm data: ", stacked_lstm_enc_data.shape)
    # write to file
    flattened_lstm_data = np.ravel(stacked_lstm_enc_data)
    print("First element: ", flattened_lstm_data[0])
    print("Second element: ", flattened_lstm_data[1])
    print("Third element: ", flattened_lstm_data[2])
    os.makedirs(os.path.dirname(enc_lstm_data_filename), exist_ok=True)
    np.savetxt(enc_lstm_data_filename, flattened_lstm_data, fmt='%1.10f')

