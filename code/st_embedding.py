from data_loader import DataLoader
import numpy as np
np.random.seed(1)
from skip_thought import SkipThoughtEncoder
import os
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
import training

start_time = time()
LOAD_STORED_VAL_EMBEDDINGS = True
enc_val_data_filename = '../data/ours/enc_val_data10_context.csv'

# load validation data
val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2 = DataLoader.load_validation_data()
number_of_validation_samples = len(val_right_ending_nr)
print("There are ", number_of_validation_samples, " validation samples.")

val_context_sentences = list(map(lambda i: [" ".join(i)], val_context_sentences))
val_context_sentences = np.array(val_context_sentences)
print("shape of context: ", val_context_sentences.shape)

if LOAD_STORED_VAL_EMBEDDINGS:
    # load skip-thought embedded validation data from file
    val_data = pd.read_csv(enc_val_data_filename, header=None).values
    val_data = val_data.reshape(number_of_validation_samples, 3, 4800)
    enc_val_context_sentences = val_data[:, 0, :]
    print("dim of loaded enc_val_context_sentences: ", enc_val_context_sentences.shape)
    enc_val_ending_sentence1 = val_data[:, 1, :]
    print("dim of loaded enc_val_ending_sentence1: ", enc_val_ending_sentence1.shape)
    enc_val_ending_sentence2 = val_data[:, 2, :]
    print("dim of loaded enc_val_ending_sentence2: ", enc_val_ending_sentence2.shape)

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


# create training and validation sets
validation_set_percentage = 0.1
random_state = 1

enc_val_ending_sentence1_trn, enc_val_ending_sentence1_val = train_test_split(enc_val_ending_sentence1,
                                                                              random_state=random_state,
                                                                              test_size=validation_set_percentage)

enc_val_ending_sentence2_trn, enc_val_ending_sentence2_val = train_test_split(enc_val_ending_sentence2,
                                                                              random_state=random_state,
                                                                              test_size=validation_set_percentage)

val_right_ending_nr_trn, val_right_ending_nr_val = train_test_split(val_right_ending_nr, random_state=random_state,
                                                                    test_size=validation_set_percentage)

enc_val_FC_context_trn, enc_val_FC_context_val = train_test_split(enc_val_context_sentences, random_state=random_state,
                                                                  test_size=validation_set_percentage)

# code to find out what's a good choice for number of epochs:
accuracies = []
# eps = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
eps = [18]
hidden_layers = 3

for epochs in eps:
    # accuracy_nc = training.runNC(first_endings=enc_val_ending_sentence1, second_endings=enc_val_ending_sentence2,
    #                           right_endings=val_right_ending_nr, hidden_layers=hidden_layers, epochs=epochs)
    # print("Accuracy NC for ", epochs, " epochs: ", accuracy_nc)
    # accuracies.append(accuracy_nc)
    accuracy_fc = training.runNN(first_endings=enc_val_ending_sentence1_trn, first_endings_val=enc_val_ending_sentence1_val,
                             second_endings=enc_val_ending_sentence2_trn, second_endings_val=enc_val_ending_sentence2_val,
                             right_endings=val_right_ending_nr_trn, right_endings_val=val_right_ending_nr_val,
                             context_sentences=enc_val_FC_context_trn, context_sentences_val=enc_val_FC_context_val,
                             hidden_layers=hidden_layers, epochs=epochs)
    print("Accuracy FC for ", epochs, " epochs: ", accuracy_fc)
    accuracies.append(accuracy_fc)

print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")
print("ACCURACY SUMMARY: ")
print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")
for i in range(len(accuracies)):
    print("Accuracy: ", accuracies[i], " with Epochs: ", eps[i])
print("-------------------------------------------------------------------------------")
print("-------------------------------------------------------------------------------")
print("Number of epochs: ", eps)
print("Accuracies: ", accuracies)
print("-------------------------------------------------------------------------------")
print("Done. It took ", (time()-start_time)/60/60, " hours. ")
