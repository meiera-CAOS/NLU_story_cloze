from nltk.tokenize import word_tokenize
from time import time

from data_loader import DataLoader

import numpy as np

np.random.seed(1)

start_time = time()

# to only load validation data
def load_validation_data():
    val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2 = DataLoader.load_validation_data()
    number_of_validation_samples = len(val_right_ending_nr)
    print("There are ", number_of_validation_samples, " validation samples.")
    return_list = []
    return_list.append(val_right_ending_nr)
    return_list.append(val_context_sentences)
    return_list.append(val_ending_sentence1)
    return_list.append(val_ending_sentence2)
    return return_list

# to only load training data
def load_training_data():
    train_context_sentences, train_ending_sentence, _ = DataLoader.load_training_data()
    number_of_training_samples = len(train_ending_sentence)
    print("There are ", number_of_training_samples, " training samples.")
    return_list = []
    return_list.append(train_context_sentences)
    return_list.append(train_ending_sentence)
    return return_list


# to load both training and validation data
def load_data(val_only=False, train_only=False):

    if val_only and train_only:
        raise Exception("val_only and train_only should not both be set to true. set them to false if you need both (or use default). ")

    if val_only:
        # load validation data only
        return load_validation_data()

    elif train_only:
        # load training data only (story title would also be available _)
        return load_training_data()

    else:
        # load both:

        # load validation data (packed: val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2)
        val_data = load_validation_data()

        # load training data (packed: train_context_sentences, train_ending_sentence)
        train_data = load_training_data()

        # return both in a tuple
        return train_data, val_data


def tokenize(data):
    func = lambda s: s[:1].lower() + s[1:] if s else ''

    # for every list in data, tokenize
    return_data = []
    for d in range(len(data)):
        return_data.append(list(map(lambda i: list(map(lambda k: word_tokenize(func(k)), i)), data[d])))

    return return_data


def create_vocabulary(data):

    # create a flattened array from all lists in data, needed to create vocab and count occurrences
    data = [word for array in data for sentence in array for word in sentence]

    # count word occurrences
    words = {}
    for i in range(len(data)):
        list = data[i]
        for w in list:
            if w in words:
                words[w] += 1
            else:
                words[w] = 1

    print("number of words: ", len(words.keys()))

    # order words by frequency and create vocab_list and vocab_dict
    i = 0
    vocab_list = []
    vocab_dict = {}
    for w in sorted(words, key=words.get, reverse=True):
        vocab_list.append(w)
        vocab_dict[w] = i
        i += 1

    vocab_list.append('<eos>')
    vocab_dict['<eos>'] = i
    i += 1
    vocab_list.append('<pad>')
    vocab_dict['<pad>'] = i

    return vocab_list, vocab_dict


# pad up to max_len with <pad>
def add_pad(y, max_len):
    num_pads = max_len-len(y)
    padding = np.repeat('<pad>', num_pads)
    y = np.concatenate((y, padding))
    return y


def add_eos_and_pad(data):
    # find sentence with maximum length
    max_len = max([len(sentence) for s_list in data for s in s_list for sentence in s]) + 1  # +1 because of <eos> that will be added
    print("max sentence length: ", max_len)

    # define lambda function to append <eos> to all sentences
    add_eos = lambda x: np.append(x, '<eos>')

    # for each sentence array d in data: for each sentence, add eos and pad with <pad> if necessary
    return_data = []
    for i in range(len(data)):
        return_data.append(list(map(lambda i: list(map(lambda k: add_pad(add_eos(k), max_len=max_len), i)), data[i])))

    return return_data


# preprocesses val data only
def preprocess_val_only():
    val_data = load_data(val_only=True)
    print("val_data shape: ", len(val_data))
    right_endings = val_data[0]
    to_tokenize = val_data[1:]
    tokenized = tokenize(to_tokenize)
    vocab_list, vocab_dict = create_vocabulary(tokenized)
    preprocessed_data = add_eos_and_pad(tokenized)
    preprocessed_val_context_sentences = preprocessed_data[0]
    preprocessed_val_endings1 = preprocessed_data[1]
    preprocessed_val_endings2 = preprocessed_data[2]
    print("prep. context: ", preprocessed_val_context_sentences[0])
    print("prep. endings1: ", preprocessed_val_endings1[0])
    print("prep. endings2: ", preprocessed_val_endings2[0])
    print("vocab_list: ", vocab_list)
    print("vocab_dict: ", vocab_dict)
    return vocab_list, vocab_dict, right_endings, preprocessed_val_context_sentences, preprocessed_val_endings1, preprocessed_val_endings2


# preprocesses train data only
def preprocess_train_only():
    train_data = load_data(train_only=True)
    print("train_data shape: ", len(train_data))
    tokenized = tokenize(train_data)
    vocab_list, vocab_dict = create_vocabulary(tokenized)
    preprocessed_data = add_eos_and_pad(tokenized)
    preprocessed_train_context_sentences = preprocessed_data[0]
    preprocessed_train_endings = preprocessed_data[1]
    print("prep. context: ", preprocessed_train_context_sentences[0])
    print("prep. endings1: ", preprocessed_train_endings[0])
    print("vocab_list: ", vocab_list)
    print("vocab_dict: ", vocab_dict)
    return vocab_list, vocab_dict, preprocessed_train_context_sentences, preprocessed_train_endings


# preprocesses val and train data
# returns: vocab_list, vocab_dict, val_data, train_data
def preprocess_all_data():
    data = load_data()
    print("data length: ", len(data))
    # data is a tuple of (train_data, val_data)
    train_data, val_data = data
    right_endings = val_data[0]
    val_to_tokenize = val_data[1:]
    to_tokenize = []
    # make list of all data to tokenize
    to_tokenize.extend(train_data + val_to_tokenize)
    print("to_tokenize shape: ", len(to_tokenize))
    print("val_data length: ", len(val_data))
    print("train_data length: ", len(train_data))
    tokenized = tokenize(to_tokenize)
    vocab_list, vocab_dict = create_vocabulary(tokenized)
    preprocessed_data = add_eos_and_pad(tokenized)
    preprocessed_train_context_sentences = preprocessed_data[0]
    preprocessed_train_endings = preprocessed_data[1]
    preprocessed_val_context_sentences = preprocessed_data[2]
    preprocessed_val_endings1 = preprocessed_data[3]
    preprocessed_val_endings2 = preprocessed_data[4]
    print("prep. context: ", preprocessed_val_context_sentences[0])
    print("prep. endings1: ", preprocessed_val_endings1[0])
    print("prep. endings2: ", preprocessed_val_endings2[0])
    print("vocab_list[:4]: ", vocab_list[:4])
    print("vocab_dict[:4]: ", vocab_dict[vocab_list[0]], vocab_dict[vocab_list[1]], vocab_dict[vocab_list[2]], vocab_dict[vocab_list[3]],)
    return vocab_list, vocab_dict, \
            (right_endings, preprocessed_val_context_sentences, preprocessed_val_endings1, preprocessed_val_endings2), \
            (preprocessed_train_context_sentences, preprocessed_train_endings)


# how to use the code:

# # to load and preprocess only the validation data:
# val_vocab_list, val_vocab_dict, val_right_endings, preprocessed_val_context_sentences, preprocessed_val_endings1, preprocessed_val_endings2 = preprocess_val_only()
# print("Val only done. ")
#
# # to load and preprocess only the training data:
# train_vocab_list, train_vocab_dict, preprocessed_train_context_sentences, preprocessed_train_endings = preprocess_train_only()
# print("Train only done. ")
#
# # to load and preprocess both val and train data (creates "shared" vocab of all data):
# vocab_list, vocab_dict, val_data, train_data = preprocess_all_data()
# right_endings, preprocessed_val_context_sentences, preprocessed_val_endings1, preprocessed_val_endings2 = val_data
# preprocessed_train_context_sentences, preprocessed_train_endings = train_data
# print("All data done. ")