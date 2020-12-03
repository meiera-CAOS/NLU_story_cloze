from data_loader import DataLoader
from time import time

import training

import numpy as np
import pandas as pd

start_time = time()

# set parameters

enc_trn_data_filename = '../data/ours/enc_val_data10.csv'
enc_trn_context_data_filename = '../data/ours/enc_val_data10_context.csv'

enc_test_data_filename = '../data/ours/enc_testpredict_data10.csv'
enc_test_context_data_filename = '../data/ours/enc_testpredict_data10_context.csv'

prediction_filename = '../data/ours/test_predictions.csv'

hidden_layers = 3
epochs = 65
with_dropout = True

# load training data to get the number of training samples (use validation set as training set)
trn_right_ending_nr, _, _, _ = DataLoader.load_validation_data()
number_of_training_samples = len(trn_right_ending_nr)
print("There are ", number_of_training_samples, " training samples.")

# load test data to get the number of test samples
_, test_ending1, _ = DataLoader.load_test_data_to_make_predictions()
number_of_test_samples = len(test_ending1)
print("There are ", number_of_test_samples, " test samples.")


# LOAD DATA
# load skip-thought embedded train data from file
trn_data = pd.read_csv(enc_trn_data_filename, header=None).values
trn_data = trn_data.reshape(number_of_training_samples, 6, 4800)
enc_trn_context_sentences = trn_data[:, :4, :]
print("dim of loaded enc_trn_context_sentences: ", enc_trn_context_sentences.shape)
enc_trn_ending_sentence1 = trn_data[:, 4, :]
print("dim of loaded enc_trn_ending_sentence1: ", enc_trn_ending_sentence1.shape)
enc_trn_ending_sentence2 = trn_data[:, 5, :]
print("dim of loaded enc_trn_ending_sentence2: ", enc_trn_ending_sentence2.shape)

trn_context_data = pd.read_csv(enc_trn_context_data_filename, header=None).values
trn_context_data = trn_context_data.reshape(number_of_training_samples, 3, 4800)
enc_trn_context_embeddings = trn_context_data[:, 0, :]

# load skip-thought embedded test data from file
test_data = pd.read_csv(enc_test_data_filename, header=None).values
test_data = test_data.reshape(number_of_test_samples, 6, 4800)
enc_test_context_sentences = test_data[:, :4, :]
print("dim of loaded enc_test_context_sentences: ", enc_test_context_sentences.shape)
enc_test_ending_sentence1 = test_data[:, 4, :]
print("dim of loaded enc_test_ending_sentence1: ", enc_test_ending_sentence1.shape)
enc_test_ending_sentence2 = test_data[:, 5, :]
print("dim of loaded enc_test_ending_sentence2: ", enc_test_ending_sentence2.shape)

test_context_data = pd.read_csv(enc_test_context_data_filename, header=None).values
test_context_data = test_context_data.reshape(number_of_test_samples, 3, 4800)
enc_test_context_embeddings = test_context_data[:, 0, :]


# TRAINING

# create contexts
FC_trn_context = enc_trn_context_embeddings
FC_test_context = enc_test_context_embeddings

# FC
model, scaler = training.create_and_train_model(first_endings=enc_trn_ending_sentence1,
                                                second_endings=enc_trn_ending_sentence2,
                                                right_endings=trn_right_ending_nr,
                                                context_sentences=FC_trn_context, hidden_layers=hidden_layers,
                                                epochs=epochs, with_dropout=with_dropout)

predicted_right_endings = training.predict_right_ending(model=model, context_sentences=FC_test_context,
                                                        first_endings=enc_test_ending_sentence1,
                                                        second_endings=enc_test_ending_sentence2, scaler=scaler)

# write the predicted ending to file
np.savetxt(prediction_filename, predicted_right_endings, fmt='%d')

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("FC with dropout and ", epochs, " epochs on test set.")
print("Done. It took ", (time()-start_time)/60/60, " hours. ")

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

