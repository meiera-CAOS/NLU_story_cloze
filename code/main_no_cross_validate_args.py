from data_loader import DataLoader
from time import time
from sklearn.utils import shuffle
import sys

import training

import numpy as np
import pandas as pd

# Takes 2 arguments of type int:
# 1) number of epochs to train (for example 55)
# 2) (optional) random seed to shuffle data, such that the same validation set can be used for different experiments
# if no random seed argument is passed, a random seed is generated

start_time = time()

# set parameters

enc_val_data_filename = '../data/ours/enc_val_data10.csv'
enc_val_context_data_filename = '../data/ours/enc_val_data10_context.csv'

enc_trn_data_filename = '../data/ours/enc_lstmfakeendings_data10_5000.csv'
enc_trn_context_data_filename = '../data/ours/enc_lstmfakeendings_data10_context_5000.csv'

real_val_percentage = 0.1        # determines how much of the validation samples will really be treated as validation data

hidden_layers = 3
# first argument is epoch number. only one epoch number can be passed!
if len(sys.argv) < 2:
    raise Exception("You need to pass at least 1 argument: number of epochs (for example 55)")

eps = [int(sys.argv[1])]
with_dropout = False


# load validation data to get the number of validation samples
val_right_ending_nr, _, _, _ = DataLoader.load_validation_data()
number_of_validation_samples = len(val_right_ending_nr)
print("There are ", number_of_validation_samples, " validation samples.")

# get the number of training samples from the file name
# find the last occurrence of '_'
start_index = enc_trn_data_filename.rfind('_')
# find the last occurrence of '.csv'
end_index = enc_trn_data_filename.rfind('.csv')

if start_index < end_index < len(enc_trn_data_filename):
    number_of_training_samples = int(enc_trn_data_filename[start_index+1:end_index])
else:
    raise Exception(
        'main: the embedded training data must contain the number of samples in its file name, but was: {}'.format(enc_trn_data_filename))

# create right ending vector for training data (simply always 1, since the fake ending appended after the real one)
trn_right_ending_nr = np.ones((number_of_training_samples, 1), dtype=int)

# LOAD DATA
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

# load skip-thought embedded train data (with fake endings) from file
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

# redistribute the validation data according to the given percentage of real validation data
real_number_of_validation_samples = int(real_val_percentage * number_of_validation_samples)
if real_number_of_validation_samples < number_of_validation_samples:
    # shuffle the validation data with the same random seed
    if len(sys.argv) > 2:
        # take the random seed that was given as an argument
        random_seed = int(sys.argv[2])
    else:
        random_seed = np.random.randint(1, 101)
    enc_val_context_sentences = shuffle(enc_val_context_sentences, random_state=random_seed)
    enc_val_ending_sentence1 = shuffle(enc_val_ending_sentence1, random_state=random_seed)
    enc_val_ending_sentence2 = shuffle(enc_val_ending_sentence2, random_state=random_seed)
    enc_val_context_embeddings = shuffle(enc_val_context_embeddings, random_state=random_seed)
    val_right_ending_nr = shuffle(val_right_ending_nr, random_state=random_seed)

    # take the first real_number_of_validation_samples as validation data and append the rest to training data
    enc_trn_context_sentences = np.concatenate((enc_trn_context_sentences, enc_val_context_sentences[real_number_of_validation_samples:]))
    enc_trn_ending_sentence1 = np.concatenate((enc_trn_ending_sentence1, enc_val_ending_sentence1[real_number_of_validation_samples:]))
    enc_trn_ending_sentence2 = np.concatenate((enc_trn_ending_sentence2, enc_val_ending_sentence2[real_number_of_validation_samples:]))
    enc_trn_context_embeddings = np.concatenate((enc_trn_context_embeddings, enc_val_context_embeddings[real_number_of_validation_samples:]))
    trn_right_ending_nr = np.concatenate((trn_right_ending_nr, val_right_ending_nr[real_number_of_validation_samples:]))

    enc_val_context_sentences = enc_val_context_sentences[:real_number_of_validation_samples]
    enc_val_ending_sentence1 = enc_val_ending_sentence1[:real_number_of_validation_samples]
    enc_val_ending_sentence2 = enc_val_ending_sentence2[:real_number_of_validation_samples]
    enc_val_context_embeddings = enc_val_context_embeddings[:real_number_of_validation_samples]
    val_right_ending_nr = val_right_ending_nr[:real_number_of_validation_samples]

    # update the number of samples
    number_of_training_samples += (number_of_validation_samples - real_number_of_validation_samples)
    number_of_validation_samples = real_number_of_validation_samples


# TRAINING

# create contexts
NC_val_context = np.zeros((number_of_validation_samples, 4800))
LS_val_context = enc_val_context_sentences[:, 3]
FC_val_context = enc_val_context_embeddings

NC_trn_context = np.zeros((number_of_training_samples, 4800))
LS_trn_context = enc_trn_context_sentences[:, 3]
FC_trn_context = enc_trn_context_embeddings

accuracies_NC = []
accuracies_LS = []
accuracies_FC = []

for epochs in eps:

    # NC
    accuracy_nc = training.runNN(first_endings=enc_trn_ending_sentence1, first_endings_val=enc_val_ending_sentence1,
                                 second_endings=enc_trn_ending_sentence2, second_endings_val=enc_val_ending_sentence2,
                                 right_endings=trn_right_ending_nr, right_endings_val=val_right_ending_nr,
                                 context_sentences=NC_trn_context, context_sentences_val=NC_val_context,
                                 hidden_layers=hidden_layers, epochs=epochs, with_dropout=with_dropout)
    accuracies_NC.append(accuracy_nc)

    # LS
    accuracy_ls = training.runNN(first_endings=enc_trn_ending_sentence1, first_endings_val=enc_val_ending_sentence1,
                                 second_endings=enc_trn_ending_sentence2, second_endings_val=enc_val_ending_sentence2,
                                 right_endings=trn_right_ending_nr, right_endings_val=val_right_ending_nr,
                                 context_sentences=LS_trn_context, context_sentences_val=LS_val_context,
                                 hidden_layers=hidden_layers, epochs=epochs, with_dropout=with_dropout)
    accuracies_LS.append(accuracy_ls)

    # FC
    accuracy_FC = training.runNN(first_endings=enc_trn_ending_sentence1, first_endings_val=enc_val_ending_sentence1,
                                 second_endings=enc_trn_ending_sentence2, second_endings_val=enc_val_ending_sentence2,
                                 right_endings=trn_right_ending_nr, right_endings_val=val_right_ending_nr,
                                 context_sentences=FC_trn_context, context_sentences_val=FC_val_context,
                                 hidden_layers=hidden_layers, epochs=epochs, with_dropout=with_dropout)
    accuracies_FC.append(accuracy_FC)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("accuracies NC: ", accuracies_NC)
print("accuracies LS: ", accuracies_LS)
print("accuracies FC: ", accuracies_FC)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")

print("best epoch number for NC: ", eps[np.argmax(accuracies_NC)])
print("best epoch number for LS: ", eps[np.argmax(accuracies_LS)])
print("best epoch number for FC: ", eps[np.argmax(accuracies_FC)])

print("-------------------------------------------------------------------------------")
print("Random seed for shuffling: ", random_seed)
print("With_dropout: ", with_dropout)
print("Real val percentage: ", real_val_percentage)
print("-------------------------------------------------------------------------------")

print("Done. It took ", (time()-start_time)/60/60, " hours. ")
