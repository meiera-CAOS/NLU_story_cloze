from data_loader import DataLoader
from time import time
from sklearn.metrics import accuracy_score

import training
import sys

import numpy as np
import pandas as pd

# Takes 1 argument: number of epochs to train (for example 55)

start_time = time()

# set parameters

enc_trn_data_filename = '../data/ours/enc_val_data10.csv'
enc_trn_context_data_filename = '../data/ours/enc_val_data10_context.csv'

enc_test_data_filename = '../data/ours/enc_testreport_data10.csv'
enc_test_context_data_filename = '../data/ours/enc_testreport_data10_context.csv'

hidden_layers = 3

if len(sys.argv) < 2:
    raise Exception("You need to pass 1 argument: number of epochs (for example 55)")

epochs = int(sys.argv[1])

# load training data to get the number of training samples (use validation set as training set)
trn_right_ending_nr, _, _, _ = DataLoader.load_validation_data()
number_of_training_samples = len(trn_right_ending_nr)
print("There are ", number_of_training_samples, " training samples.")

# load test data to get the number of test samples
test_right_ending_nr, test_context_sentences, test_ending_sentence1, test_ending_sentence2 = DataLoader.load_test_data_with_right_ending_nr()
number_of_test_samples = len(test_right_ending_nr)
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

accuracies_dropout = []
accuracies_normal = []
for repetition in range(3):

    # FC with dropout
    model, scaler = training.create_and_train_model(first_endings=enc_trn_ending_sentence1,
                                                    second_endings=enc_trn_ending_sentence2,
                                                    right_endings=trn_right_ending_nr,
                                                    context_sentences=FC_trn_context, hidden_layers=hidden_layers,
                                                    epochs=epochs, with_dropout=True)

    predicted_right_endings = training.predict_right_ending(model=model, context_sentences=FC_test_context,
                                                            first_endings=enc_test_ending_sentence1,
                                                            second_endings=enc_test_ending_sentence2, scaler=scaler)

    accuracies_dropout.append(accuracy_score(y_true=test_right_ending_nr, y_pred=predicted_right_endings))

    # FC no dropout
    model, scaler = training.create_and_train_model(first_endings=enc_trn_ending_sentence1,
                                                    second_endings=enc_trn_ending_sentence2,
                                                    right_endings=trn_right_ending_nr,
                                                    context_sentences=FC_trn_context, hidden_layers=hidden_layers,
                                                    epochs=epochs, with_dropout=False)

    predicted_right_endings = training.predict_right_ending(model=model, context_sentences=FC_test_context,
                                                            first_endings=enc_test_ending_sentence1,
                                                            second_endings=enc_test_ending_sentence2, scaler=scaler)

    accuracies_normal.append(accuracy_score(y_true=test_right_ending_nr, y_pred=predicted_right_endings))


print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("FC with ", epochs, " epochs on testreport set.")
print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("Accuracies dropout: ", accuracies_dropout)
print("Average accuracy dropout: ", np.mean(accuracies_dropout))

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("Accuracies normal: ", accuracies_normal)
print("Average accuracy normal: ", np.mean(accuracies_normal))

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("Done. It took ", (time()-start_time)/60, " minutes. ")
