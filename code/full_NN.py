import tensorflow
from tensorflow import keras
import numpy as np
from tensorflow.python.keras.constraints import max_norm

np.random.seed(1)
tensorflow.set_random_seed(1)


# x: number_of_training_samples x 4800, skipthought sum of ending and context
# y: number_of_training_samples x 1, label 0 if wrong and 1 if right ending.
# hidden_layers: 2 or 3, depending on wanted number of hidden layers in neural network
# epochs: number of epochs
# returns: the trained model
def build_nn_and_train(hidden_layers, epochs, x, y, with_dropout):
    if hidden_layers == 3:
        # build model with 3 hidden layers and train
        if with_dropout:
            # create model with dropout layers and kernel_constraint
            model = keras.Sequential([
                    keras.layers.Dense(2400, input_dim=4800, kernel_constraint=max_norm(5), activation='relu'),
                    keras.layers.Dropout(rate=0.5),
                    keras.layers.Dense(1200, kernel_constraint=max_norm(5), activation='relu'),
                    keras.layers.Dropout(rate=0.5),
                    keras.layers.Dense(600, kernel_constraint=max_norm(5), activation='relu'),
                    keras.layers.Dropout(rate=0.5),
                    keras.layers.Dense(2, activation='softmax')
            ])
        else:
            # create model without dropout layers and without kernel_constraint
            model = keras.Sequential([
                keras.layers.Dense(2400, input_dim=4800, activation='relu'),
                keras.layers.Dense(1200, activation='relu'),
                keras.layers.Dense(600, activation='relu'),
                keras.layers.Dense(2, activation='softmax')
            ])
    elif hidden_layers == 2:
        # build model with 2 hidden layers and train
        model = keras.Sequential([
                keras.layers.Dense(256, input_dim=4800, activation='relu'),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(2, activation='softmax')
        ])

    else:
        raise Exception('build_nn_and_train: hidden layer argument is invalid, must be 2 or 3, but was: {}'.format(hidden_layers))

    opt = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(np.array(x), np.array(y), batch_size=32, shuffle=True, epochs=epochs, verbose=0)
    print("Model Summary: ")
    print(model.summary())
    return model


# model: trained model
# context: number_of_test_samples x 4800, skipthought vector that encodes the context (FC, NC or LS)
# ending1: number_of_test_samples x 4800, skipthought vector of the first ending sentence
# ending2: number_of_test_samples x 4800, skipthought vector of the second ending sentence
# returns: number_of_test_samples x 1, 1 if first ending is correct, 2 otherwise
# scaler: StandardScaler, if it's not None we should transform the input with it
def predict(model, context, ending1, ending2, scaler=None):

    # first ending:
    input_for_first_ending = context + ending1
    if scaler is not None:
        input_for_first_ending = scaler.transform(input_for_first_ending)
    y_first_ending_predicted = model.predict(input_for_first_ending)
    prob_first_ending_right = y_first_ending_predicted[:, 1]

    # second ending:
    input_for_second_ending = context + ending2
    if scaler is not None:
        input_for_second_ending = scaler.transform(input_for_second_ending)
    y_second_ending_predicted = model.predict(input_for_second_ending)
    prob_second_ending_right = y_second_ending_predicted[:, 1]

    # see which sentence has higher probability of being right
    stacked_probabilities = np.stack((prob_first_ending_right, prob_second_ending_right), axis=1)
    right_endings = np.argmax(stacked_probabilities, axis=1)
    # increment indices for correct output format: 1 or 2 instead of 0 or 1
    right_endings += 1
    return right_endings
