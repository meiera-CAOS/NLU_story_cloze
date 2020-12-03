from time import time

import sys
sys.path.append('../')

import lstm.model as model
import lstm.preprocessing as preprocessing
import lstm.word_embedding as word_embedding

import numpy as np
import tensorflow as tf

start_time = time()

# define parameters

# flags
GENERATE_SENTENCES = False
DISCRIMINATE_ENDINGS = True

# input preparation
number_of_timesteps = 20
number_of_skip_words = 5
word_embedding_dimension = 100

# model
epochs = 10
lstm_size = 256

# generation
max_generated_sentence_length = 30
filename_generated_sentences = "../data/ours/lstm_generated_sentences_" + str(epochs) + "epochs"


# load and pre-process the training and validation data
vocab_list, vocab_dict, val_data, train_data = preprocessing.preprocess_all_data()
val_right_endings, preprocessed_val_context_sentences, preprocessed_val_endings1, preprocessed_val_endings2 = val_data
preprocessed_train_context_sentences, preprocessed_train_endings = train_data

vocab_size = len(vocab_list)

# limit number of sentences for debug purposes
# number_of_sentences = 20
# preprocessed_train_context_sentences = preprocessed_train_context_sentences[:number_of_sentences]
# preprocessed_train_endings = preprocessed_train_endings[:number_of_sentences]

# preprocessed_val_context_sentences = preprocessed_val_context_sentences[:number_of_sentences]
# preprocessed_val_endings1 = preprocessed_val_endings1[:number_of_sentences]
# preprocessed_val_endings2 = preprocessed_val_endings2[:number_of_sentences]

# prepare training input: add endings to the context sentences, concatenate the sentences to one list and remove padding
training_input = []
for context_list, ending_list in zip(preprocessed_train_context_sentences, preprocessed_train_endings):
    training_input.append(np.concatenate((context_list, ending_list)))

training_input = list(map(lambda list_of_arrays: list(np.concatenate(list_of_arrays)), training_input))
training_input = list(map(lambda list_of_strings: list(filter(lambda token: token != '<pad>', list_of_strings)), training_input))

# load word pre-trained word embedding
session = tf.Session()
word_embedding_tensor = word_embedding.getWord2VecEmbedding(session=session, vocabularySize=vocab_size,
                                                            embeddingDimension=word_embedding_dimension, vocab=vocab_dict)

word_embedding_matrix = session.run(word_embedding_tensor)

# prepare the input (create n-grams)
x, y = model.prepare_training_input(data=training_input, number_of_timesteps=number_of_timesteps, vocab_dict=vocab_dict,
                                    number_of_skip_words=number_of_skip_words)

# define and train model
lstm_model = model.train_model(x=x, y=y, epochs=epochs, lstm_size=lstm_size, vocab_size=vocab_size,
                               embedding_dimension=word_embedding_dimension, embedding_matrix=word_embedding_matrix)

if GENERATE_SENTENCES:

    # prepare generation input: concatenate the sentences to one list and remove padding
    generation_input = preprocessed_train_context_sentences
    generation_input = list(map(lambda list_of_arrays: list(np.concatenate(list_of_arrays)), generation_input))
    generation_input = list(map(lambda list_of_strings: list(filter(lambda token: token != '<pad>', list_of_strings)), generation_input))

    # generate sentences using the model
    context = model.prepare_input(data=generation_input, number_of_timesteps=number_of_timesteps, vocab_dict=vocab_dict)
    generated_sentences = model.generate(model=lstm_model, context=context, vocab_list=vocab_list,
                                         max_generated_sentence_length=max_generated_sentence_length)

    print("Writing ", len(generated_sentences), " generated sentences to file. ")
    np.savetxt(filename_generated_sentences, generated_sentences, fmt='%s')

if DISCRIMINATE_ENDINGS:

    # prepare validation/discrimination input: concatenate the context sentences to one list and remove padding
    validation_input = preprocessed_val_context_sentences
    validation_input = list(map(lambda list_of_arrays: list(np.concatenate(list_of_arrays)), validation_input))
    validation_input = list(map(lambda list_of_strings: list(filter(lambda token: token != '<pad>', list_of_strings)), validation_input))

    # encode endings (don't remove pads) and reformat into an array (number_of_samples x max_ending_length)
    validation_ending1 = list(map(lambda list_of_arrays: list_of_arrays[0], preprocessed_val_endings1))
    encoded_validation_ending1 = np.array(list(map(lambda s: list(map(lambda w: vocab_dict[w], s)), validation_ending1)))

    validation_ending2 = list(map(lambda list_of_arrays: list_of_arrays[0], preprocessed_val_endings2))
    encoded_validation_ending2 = np.array(list(map(lambda s: list(map(lambda w: vocab_dict[w], s)), validation_ending2)))

    # create padding mask of the ending arrays
    index_of_pad = vocab_dict['<pad>']
    validation_ending1_pad_mask = (encoded_validation_ending1 != index_of_pad)
    validation_ending2_pad_mask = (encoded_validation_ending2 != index_of_pad)

    # get probabilities of the language model for both endings
    validation_context = model.prepare_input(data=validation_input, number_of_timesteps=number_of_timesteps, vocab_dict=vocab_dict)

    probabilities_ending1 = model.get_probability_for_ending(model=lstm_model, context=validation_context,
                                                             endings=encoded_validation_ending1)

    probabilities_ending2 = model.get_probability_for_ending(model=lstm_model, context=validation_context,
                                                             endings=encoded_validation_ending2)

    # get real length of ending sentences
    ending1_sentences_length = np.count_nonzero(validation_ending1_pad_mask, axis=1)
    ending2_sentences_length = np.count_nonzero(validation_ending2_pad_mask, axis=1)

    # mask out the probabilities of the padding
    probabilities_ending1 *= validation_ending1_pad_mask
    probabilities_ending2 *= validation_ending2_pad_mask

    # compute average probabilities of both endings
    summed_probabilities_ending1 = np.sum(probabilities_ending1, axis=1)
    summed_probabilities_ending2 = np.sum(probabilities_ending2, axis=1)

    averaged_probabilities_ending1 = summed_probabilities_ending1 / ending1_sentences_length
    averaged_probabilities_ending2 = summed_probabilities_ending2 / ending2_sentences_length

    predicted_correct_ending = np.argmax(np.stack((averaged_probabilities_ending1, averaged_probabilities_ending2), axis=1), axis=1) + 1

    correct_predictions = np.count_nonzero(np.diag(val_right_endings == predicted_correct_ending))
    number_of_validation_samples = len(val_right_endings)
    accuracy = correct_predictions / number_of_validation_samples

    print("Accuracy of the language model with ", epochs, " epochs was: ", accuracy)


print("Done. It took ", (time() - start_time) / 60, " minutes for ", epochs, " epochs.")

