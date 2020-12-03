from skip_thought import SkipThoughtEncoder
from data_loader import DataLoader
import numpy as np
np.random.seed(1)
import os
from time import time
import sys

# TO CREATE SKIPTHOUGHT EMBEDDINGS OF GAN-GENERATED TRAINING DATA
# Takes 1 optional argument: number of samples to create the embedding for. If no argument, it encodes all samples.

cgan_generated_file_name = "../data/ours/GAN_sentences_POSTPROCESSED"
start_time = time()
LOAD_STORED_VAL_EMBEDDINGS = False
enc_cgan_data_filename = "../data/ours/enc_cganfakeendings_data10_" + str(sys.argv[1]) + ".csv"
enc_cgan_context_data_filename = "../data/ours/enc_cganfakeendings_data10_context_" + str(sys.argv[1]) + ".csv"

# load NEW TRAINING data with GENERATED FAKE ENDINGS
d = DataLoader()
cgan_right_ending_nr, cgan_context_sentences, cgan_ending_sentence1, cgan_ending_sentence2 = d.load_data_with_fake_endings(file_name=cgan_generated_file_name)
number_of_training_samples = len(cgan_right_ending_nr)
print("There are ", number_of_training_samples, " training samples.")

if not len(sys.argv) == 1:
    n_samples = int(sys.argv[1])
    cgan_right_ending_nr = cgan_right_ending_nr[:n_samples]
    cgan_context_sentences = cgan_context_sentences[:n_samples]
    cgan_ending_sentence1 = cgan_ending_sentence1[:n_samples]
    cgan_ending_sentence2 = cgan_ending_sentence2[:n_samples]

print("shape of trn context: ", cgan_context_sentences.shape)

if LOAD_STORED_VAL_EMBEDDINGS:
    print("Do nothing. ")

else:
    # initialize SkipThoughtEncoder
    s = SkipThoughtEncoder()
    s.init()

    # skipthought-encode cgan generated data
    cgan_enc_ending_sentence2 = np.array(list(map(s.encoder.encode, cgan_ending_sentence2)))
    cgan_enc_context_sentences = np.array(list(map(s.encoder.encode, cgan_context_sentences)))
    print("Dimension of cgan context vector: ", cgan_enc_context_sentences.shape)
    cgan_enc_ending_sentence1 = np.array(list(map(s.encoder.encode, cgan_ending_sentence1)))

    # stack encoded cgan data to write to file
    stacked_endings = np.concatenate((cgan_enc_ending_sentence1, cgan_enc_ending_sentence2), axis=1)
    print("shape of cgan_enc_context_sentences: ", cgan_enc_context_sentences.shape)
    print("shape of stacked_endings: ", stacked_endings.shape)
    stacked_cgan_enc_data = np.concatenate((cgan_enc_context_sentences, stacked_endings), axis=1)

    # expected data shape: number_of_cgan_samples x 6 x 4800
    print("Dimension of stacked cgan data: ", stacked_cgan_enc_data.shape)
    # write to file
    flattened_cgan_data = np.ravel(stacked_cgan_enc_data)
    print("First element: ", flattened_cgan_data[0])
    print("Second element: ", flattened_cgan_data[1])
    print("Third element: ", flattened_cgan_data[2])
    os.makedirs(os.path.dirname(enc_cgan_data_filename), exist_ok=True)
    np.savetxt(enc_cgan_data_filename, flattened_cgan_data, fmt='%1.10f')

