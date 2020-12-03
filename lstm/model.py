from tensorflow import keras

import numpy as np


# x: number_of_training_samples x number_of_timesteps, one-hot-encoded (word ids) lstm input
# y: number_of_training_samples, one-hot-encoded (word ids) label
# epochs: number of epochs
# lstm_size: hidden state size of the lstm
# vocab_size: size of the used vocabulary
# embedding_dimension: the embedding dimension of the given embedding matrix
# embedding_matrix: vocab_size x embedding_dimension, pre-trained embedding matrix used as initial weight matrix of the embedding layer
def train_model(x, y, epochs, lstm_size, vocab_size, embedding_dimension, embedding_matrix):
    # convert x and y to tensors
    # build model
    lstm_model = keras.Sequential([
            keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dimension, weights=[embedding_matrix],
                                   trainable=False),
            keras.layers.LSTM(lstm_size),
            keras.layers.Dense(vocab_size, activation='softmax')
    ])

    # define model parameters
    optimizer = 'adam'                          # keras.optimizers.Adam(lr=0.001)
    loss = 'sparse_categorical_crossentropy'    # 'categorical_crossentropy'
    metrics = ['sparse_categorical_crossentropy', 'sparse_categorical_accuracy']

    lstm_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # train the model
    lstm_model.fit(x, y, batch_size=64, shuffle=True, epochs=epochs, verbose=1)
    print("Model Summary: ")
    print(lstm_model.summary())
    return lstm_model


# model: the pre-trained model used to generate sentences
# context: number_of_samples x num_of_timesteps, one-hot-encoded context based on which the prediction runs
# max_generated_sentence_length: maximum length of a generated sentence (stop criteria in case no <eos> was generated)
def generate(model, context, vocab_list, max_generated_sentence_length):
    number_of_samples = len(context)
    generated_word_ids = np.zeros((number_of_samples, max_generated_sentence_length), dtype=int)

    for i in range(max_generated_sentence_length):
        predicted_probabilities = model.predict(context)
        predicted_word_ids = np.argmax(predicted_probabilities, axis=1)

        # store the generated words
        generated_word_ids[:, i] = predicted_word_ids

        # create the next prediction input by removing first column of the context and adding the predicted words
        predicted_word_ids = np.reshape(predicted_word_ids, (number_of_samples, 1))
        context = np.concatenate((context[:, 1:], predicted_word_ids), axis=1)

    # post-process the generated words to sentences until but without <eos> tag
    generated_sentences = []
    for i in range(number_of_samples):
        sentence = []
        for j in range(max_generated_sentence_length):
            word = vocab_list[generated_word_ids[i][j]]
            if word != '<eos>':
                sentence.append(word)
            else:
                break
        generated_sentences.append(' '.join(sentence))

    return generated_sentences


# model: the pre-trained model used to generate sentences
# context: number_of_samples x num_of_timesteps, one-hot-encoded context based on which the prediction runs
# ending: number_of_samples x max_ending_length, the ending for which the probabilities should be computed
def get_probability_for_ending(model, context, endings):
    number_of_samples, max_ending_length = endings.shape
    probabilities = np.zeros((number_of_samples, max_ending_length), dtype=float)

    for j in range(max_ending_length):
        predicted_probabilities = model.predict(context)

        for i in range(number_of_samples):
            probabilities[i][j] = predicted_probabilities[i][endings[i][j]]

        # create the next prediction input by removing first column of the context and adding the next word of ending
        ending_next_word = np.reshape(endings[:, j], (number_of_samples, 1))
        context = np.concatenate((context[:, 1:], ending_next_word), axis=1)

    return probabilities


# data: list (length = number of samples) of lists (length = variable length of sentences) of strings
# number_of_timesteps: number of time steps used in lstm
# number_of_skip_words: number of words to be skipped when creating fixed size sequences
# vocab_dict: dictionary [word -> word id]
# returns:
#           x: number_of_sequences x number_of_timesteps, one-hot-encoded input sequence for the lstm
#           y: number_of_sequences, one-hot-encoded target word for lstm
def prepare_training_input(data, number_of_timesteps, number_of_skip_words, vocab_dict):

    # create fixed size sequences x with the corresponding y
    x = []
    y = []
    too_short_sentences = 0
    shortest_sentence_length = number_of_timesteps
    for sentence in data:
        # only sentences that are longer than number_of_timesteps are considered (others are discarded)
        sentence_length = len(sentence)
        if sentence_length > number_of_timesteps:
            start_index_x = 0
            start_index_y = start_index_x + number_of_timesteps

            while start_index_y < sentence_length:
                x.append(sentence[start_index_x:start_index_y])
                y.append(sentence[start_index_y])

                start_index_x += number_of_skip_words
                start_index_y += number_of_skip_words
        else:
            too_short_sentences += 1
            if sentence_length < shortest_sentence_length:
                shortest_sentence_length = sentence_length

    # TODO: maybe throw exception instead of print warning? would be good in the case of the test samples
    if too_short_sentences > 0:
        print("WARNING: ", too_short_sentences, " sentences were discarded because they were to short. The shortest sentence has a length of ",
              shortest_sentence_length, ". Try reducing the number of time steps.")

    # encode x and y using given vocab
    encoded_x = np.array(list(map(lambda s: list(map(lambda w: vocab_dict[w], s)), x)))
    encoded_y = np.array(list(map(lambda word: vocab_dict[word], y)))

    return encoded_x, encoded_y


# data: list (length = number of samples) of lists (length = variable length of sentences) of strings
# number_of_timesteps: number of time steps used in lstm
# vocab_dict: dictionary [word -> word id]
# returns: number_of_samples x number_of_timesteps, one-hot-encoded last words of given sentence
def prepare_input(data, number_of_timesteps, vocab_dict):

    # create fixed size sequences of the last number_of_timesteps words
    x = []
    too_short_sentences = 0
    shortest_sentence_length = number_of_timesteps
    for sentence in data:
        # only sentences that are longer than number_of_timesteps are considered (others are discarded)
        sentence_length = len(sentence)
        if sentence_length > number_of_timesteps:
            x.append(sentence[-number_of_timesteps:])
        else:
            too_short_sentences += 1
            if sentence_length < shortest_sentence_length:
                shortest_sentence_length = sentence_length

    # TODO: maybe throw exception instead of print warning? would be good in the case of the test samples
    if too_short_sentences > 0:
        print("WARNING: ", too_short_sentences, " sentences were discarded because they were to short. The shortest sentence has a length of ",
              shortest_sentence_length, ". Try reducing the number of time steps.")

    # encode x using given vocab
    encoded_x = np.array(list(map(lambda s: list(map(lambda w: vocab_dict[w], s)), x)))

    return encoded_x
