from data_loader import DataLoader
import numpy as np
np.random.seed(1)
from skip_thought import SkipThoughtEncoder
import os
from time import time

# TO CREATE SKIPTHOUGHT EMBEDDINGS OF 4 CONTEXT SENTENCES OF LSTM GENERATED DATA

start_time = time()
LOAD_STORED_VAL_EMBEDDINGS = False
enc_test_data_filename = "../data/ours/enc_testpredict_data10.csv"
enc_test_context_data_filename = "../data/ours/enc_testpredict_data10_context.csv"

# load NEW TRAINING data with GENERATED FAKE ENDINGS
d = DataLoader()
test_context_sentences, test_ending_sentence1, test_ending_sentence2 = d.load_test_data_to_make_predictions()
number_of_training_samples = len(test_ending_sentence1)
print("There are ", number_of_training_samples, " test training samples.")

test_context_sentences = list(map(lambda i: [" ".join(i)], test_context_sentences))
test_context_sentences = np.array(test_context_sentences)
print("shape of test data context: ", test_context_sentences.shape)

if LOAD_STORED_VAL_EMBEDDINGS:
    print("Do nothing.")

else:
    # initialize SkipThoughtEncoder
    s = SkipThoughtEncoder()
    s.init()

    # skipthought-encode test generated data
    enc_test_ending_sentence2 = np.array(list(map(s.encoder.encode, test_ending_sentence2)))
    enc_test_context_sentences = np.array(list(map(s.encoder.encode, test_context_sentences)))
    print("Dimension of test context vector: ", enc_test_context_sentences.shape)
    enc_test_ending_sentence1 = np.array(list(map(s.encoder.encode, test_ending_sentence1)))

    # stack encoded test data to write to file
    stacked_endings = np.concatenate((enc_test_ending_sentence1, enc_test_ending_sentence2), axis=1)
    print("shape of enc_test_context_sentences: ", enc_test_context_sentences.shape)
    print("shape of stacked_endings: ", stacked_endings.shape)
    stacked_enc_test_data = np.concatenate((enc_test_context_sentences, stacked_endings), axis=1)

    # expected data shape: number_of_test_samples x 3 x 4800
    print("Dimension of stacked test data: ", stacked_enc_test_data.shape)
    # write to file
    flattened_test_data = np.ravel(stacked_enc_test_data)
    print("First element: ", flattened_test_data[0])
    print("Second element: ", flattened_test_data[1])
    print("Third element: ", flattened_test_data[2])

    os.makedirs(os.path.dirname(enc_test_context_data_filename), exist_ok=True)
    np.savetxt(enc_test_context_data_filename, flattened_test_data, fmt='%1.10f')

print("Done. It took ", (time() - start_time) / 60 / 60, " hours. ")
