from skip_thought import SkipThoughtEncoder
from data_loader import DataLoader
import numpy as np
np.random.seed(1)
import os
from time import time

# TO CREATE SKIPTHOUGHT EMBEDDINGS OF LSTM-GENERATED TRAINING DATA

start_time = time()
LOAD_STORED_VAL_EMBEDDINGS = False
enc_test_data_filename = "../data/ours/enc_testpredict_data10.csv"
enc_test_context_data_filename = "../data/ours/enc_testpredict_data10_context.csv"

# load NEW TRAINING data with GENERATED FAKE ENDINGS
d = DataLoader()
test_context_sentences, test_ending_sentence1, test_ending_sentence2 = d.load_test_data_to_make_predictions()
number_of_training_samples = len(test_ending_sentence1)
print("There are ", number_of_training_samples, " training samples.")

print("shape of trn context: ", test_context_sentences.shape)

if LOAD_STORED_VAL_EMBEDDINGS:
    print("Do nothing. ")

else:
    # initialize SkipThoughtEncoder
    s = SkipThoughtEncoder()
    s.init()

    # skipthought-encode test data
    test_enc_ending_sentence2 = np.array(list(map(s.encoder.encode, test_ending_sentence2)))
    test_enc_context_sentences = np.array(list(map(s.encoder.encode, test_context_sentences)))
    print("Dimension of test context vector: ", test_enc_context_sentences.shape)
    test_enc_ending_sentence1 = np.array(list(map(s.encoder.encode, test_ending_sentence1)))

    # stack encoded test data to write to file
    stacked_endings = np.concatenate((test_enc_ending_sentence1, test_enc_ending_sentence2), axis=1)
    print("shape of test_enc_context_sentences: ", test_enc_context_sentences.shape)
    print("shape of stacked_endings: ", stacked_endings.shape)
    stacked_test_enc_data = np.concatenate((test_enc_context_sentences, stacked_endings), axis=1)

    # expected data shape: number_of_test_samples x 6 x 4800
    print("Dimension of stacked test data: ", stacked_test_enc_data.shape)
    # write to file
    flattened_test_data = np.ravel(stacked_test_enc_data)
    print("First element: ", flattened_test_data[0])
    print("Second element: ", flattened_test_data[1])
    print("Third element: ", flattened_test_data[2])
    os.makedirs(os.path.dirname(enc_test_data_filename), exist_ok=True)
    np.savetxt(enc_test_data_filename, flattened_test_data, fmt='%1.10f')

