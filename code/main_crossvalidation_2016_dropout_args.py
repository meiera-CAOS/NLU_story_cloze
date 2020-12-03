from skip_thought import SkipThoughtEncoder
from data_loader import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
np.random.seed(1)
import os
import pandas as pd
from time import time
import training
import sys

# Takes 1 argument: number of epochs to train (for example 55)

if len(sys.argv) < 2:
    raise Exception("You need to pass 1 argument: number of epochs (for example 55)")

eps = [int(sys.argv[1])]

start_time = time()
LOAD_STORED_VAL_EMBEDDINGS = True
enc_val_data_filename = '../data/ours/enc_val_data10.csv'
enc_val_context_data_filename = '../data/ours/enc_val_data10_context.csv'

# training modes: training_with_dropout: true if (with scaling and dropout) and false if normal (no dropout)
training_with_dropout = True

# load validation data
file = "../data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv"
val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2 = DataLoader.load_validation_data()
number_of_validation_samples = len(val_right_ending_nr)
print("There are ", number_of_validation_samples, " validation samples.")

# load training data (story title would also be available _)
train_context_sentences, train_ending_sentence, _ = DataLoader.load_training_data()
number_of_training_samples = len(train_ending_sentence)
print("There are ", number_of_training_samples, " training samples.")

print("shape of context: ", val_context_sentences.shape)

if LOAD_STORED_VAL_EMBEDDINGS:
    # load skip-thought embedded validation data from file
    val_data = pd.read_csv(enc_val_data_filename, header=None).values
    val_data = val_data.reshape(number_of_validation_samples, 6, 4800)
    enc_val_context_sentences = val_data[:, :4, :]
    print("dim of loaded enc_val_context_sentences: ", enc_val_context_sentences.shape)
    enc_val_ending_sentence1 = val_data[:, 4, :]
    print("dim of loaded enc_val_ending_sentence1: ", enc_val_ending_sentence1.shape)
    enc_val_ending_sentence2 = val_data[:, 5, :]
    print("dim of loaded enc_val_ending_sentence2: ", enc_val_ending_sentence2.shape)

    val_context_data = pd.read_csv(enc_val_context_data_filename, header=None).values
    val_context_data = val_context_data.reshape(number_of_validation_samples, 3, 4800)
    enc_val_context_embeddings = val_context_data[:, 0, :]

else:
    # initialize SkipThoughtEncoder
    s = SkipThoughtEncoder()
    s.init()

    # skipthought-encode validation data
    enc_val_context_sentences = np.array(list(map(s.encoder.encode, val_context_sentences)))
    print("Dimension of val context vector: ", enc_val_context_sentences.shape)
    enc_val_ending_sentence1 = np.array(list(map(s.encoder.encode, val_ending_sentence1)))
    enc_val_ending_sentence2 = np.array(list(map(s.encoder.encode, val_ending_sentence2)))

    # stack encoded validation data to write to file
    stacked_endings = np.concatenate((enc_val_ending_sentence1, enc_val_ending_sentence2), axis=1)
    print("shape of enc_val_context_sentences: ", enc_val_context_sentences.shape)
    print("shape of stacked_endings: ", stacked_endings.shape)
    stacked_enc_val_data = np.concatenate((enc_val_context_sentences, stacked_endings), axis=1)

    # expected data shape: number_of_validation_samples x 6 x 4800
    print("Dimension of stacked val data: ", stacked_enc_val_data.shape)
    # write to file
    flattened_val_data = np.ravel(stacked_enc_val_data)
    print("First element: ", flattened_val_data[0])
    print("Second element: ", flattened_val_data[1])
    print("Third element: ", flattened_val_data[2])
    os.makedirs(os.path.dirname(enc_val_data_filename), exist_ok=True)
    np.savetxt(enc_val_data_filename, flattened_val_data, fmt='%1.10f')

    val_context_sentences = list(map(lambda i: [" ".join(i)], val_context_sentences))
    val_context_sentences = np.array(val_context_sentences)
    enc_val_context_embeddings = np.array(list(map(s.encoder.encode, val_context_sentences)))

# training
hidden_layers = 3

# limit the number of sentences used
n_sentences = number_of_validation_samples
enc_val_ending_sentence1 = enc_val_ending_sentence1[:n_sentences]
enc_val_ending_sentence2 = enc_val_ending_sentence2[:n_sentences]
enc_val_context_sentences = enc_val_context_sentences[:n_sentences]
val_right_ending_nr = val_right_ending_nr[:n_sentences]

# create contexts
NC_context = np.zeros((number_of_validation_samples, 4800))
LS_context = enc_val_context_sentences[:, 3]
FC_context = enc_val_context_embeddings

accuracies_NC = []
accuracies_LS = []
accuracies_FC = []

for epochs in eps:

    print("Doing epoch number: ", epochs)
    # cross validation NC
    cv_accuracies_NC = training.do_cross_validation(number_of_splits=10, first_endings=enc_val_ending_sentence1,
                                                    second_endings=enc_val_ending_sentence2, right_endings=val_right_ending_nr,
                                                    context_sentences=NC_context, epochs=epochs, hidden_layers=hidden_layers,
                                                    with_scaling=training_with_dropout)
    accuracies_NC.append(cv_accuracies_NC)

    # cross validation LS
    cv_accuracies_LS = training.do_cross_validation(number_of_splits=10, first_endings=enc_val_ending_sentence1,
                                                    second_endings=enc_val_ending_sentence2, right_endings=val_right_ending_nr,
                                                    context_sentences=LS_context, epochs=epochs, hidden_layers=hidden_layers,
                                                    with_scaling=training_with_dropout)
    accuracies_LS.append(cv_accuracies_LS)

    # cross validation LS
    cv_accuracies_FC = training.do_cross_validation(number_of_splits=10, first_endings=enc_val_ending_sentence1,
                                                    second_endings=enc_val_ending_sentence2, right_endings=val_right_ending_nr,
                                                    context_sentences=FC_context, epochs=epochs, hidden_layers=hidden_layers,
                                                    with_scaling=training_with_dropout)
    accuracies_FC.append(cv_accuracies_FC)

    print("Done with epoch number ", epochs, ".")

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("accuracies NC: ", accuracies_NC)
print("accuracies LS: ", accuracies_LS)
print("accuracies FC: ", accuracies_FC)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

avg_acc_NC = list(map(np.average, accuracies_NC))
avg_acc_LS = list(map(np.average, accuracies_LS))
avg_acc_FC = list(map(np.average, accuracies_FC))

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("average accuracies NC: ", avg_acc_NC)
print("average accuracies LS: ", avg_acc_LS)
print("average accuracies FC: ", avg_acc_FC)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("best epoch number for NC: ", eps[np.argmax(avg_acc_NC)])
print("best epoch number for LS: ", eps[np.argmax(avg_acc_LS)])
print("best epoch number for FC: ", eps[np.argmax(avg_acc_FC)])
print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("Done. It took ", (time()-start_time)/60/60, " hours. ")