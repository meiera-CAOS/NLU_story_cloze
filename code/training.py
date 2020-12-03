from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import numpy as np

import full_NN

np.random.seed(1)


# first_endings: number_of_samples x 4800, the skip-thought-embedded first endings (used for training the model)
# second_endings: number_of_samples x 4800, the skip-thought-embedded second endings (used for training the model)
# right_endings: number_of_samples x 1, 1 if first ending is correct, 2 otherwise (used for training the model)
# context_sentences: number_of_samples x 1, 4800, the context to be used (used for training the model)

# first_endings_val: number_of_samples x 4800, the skip-thought-embedded first endings (used for validation)
# second_endings_val: number_of_samples x 4800, the skip-thought-embedded second endings (used for validation)
# right_endings_val: number_of_samples x 1, 1 if first ending is correct, 2 otherwise (used for validation)
# context_sentences_val: number_of_samples x 1, 4800, the context to be used (used for training the model)

# hidden_layers: 2 or 3, depending on wanted number of hidden layers in neural network
# epochs: number of epochs

# context_sentences: determines implicitly the mode in which the NN is run.
# NC case: number_of_samples x 4800, filled with zeros
# LS case: number_of_samples x 4800, skip-thought embedding of last (4th) context sentence
# FC case: number_of_samples x 4800, encoding of skip-thought embeddings of all 4 context sentences
def runNN(first_endings, second_endings, right_endings, context_sentences, first_endings_val, second_endings_val,
          right_endings_val, context_sentences_val, hidden_layers, epochs, with_dropout):
    # train the model
    x, y = create_training_xy(first_endings=first_endings, second_endings=second_endings,
                              right_endings=right_endings, context_sentences=context_sentences)
    # if it is with_dropout, transform x
    scaler = None
    if with_dropout:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

    model = full_NN.build_nn_and_train(hidden_layers=hidden_layers, epochs=epochs, x=x, y=y, with_dropout=with_dropout)

    # get predictions for the validation set
    predicted_right_endings = full_NN.predict(model=model, context=context_sentences_val, ending1=first_endings_val,
                                              ending2=second_endings_val, scaler=scaler)

    accuracy = accuracy_score(y_true=right_endings_val, y_pred=predicted_right_endings)

    return accuracy


# first_endings: number_of_samples x 4800, the skip-thought-embedded first endings (used for training the model)
# second_endings: number_of_samples x 4800, the skip-thought-embedded second endings (used for training the model)
# right_endings: number_of_samples x 1, 1 if first ending is correct, 2 otherwise (used for training the model)
# context_sentences: number_of_samples x 1, 4800, the context to be used (used for training the model)
# hidden_layers: 2 or 3, depending on wanted number of hidden layers in neural network
# epochs: number of epochs
def create_and_train_model(first_endings, second_endings, right_endings, context_sentences, hidden_layers, epochs,
                           with_dropout):
    # train the model
    x, y = create_training_xy(first_endings=first_endings, second_endings=second_endings,
                              right_endings=right_endings, context_sentences=context_sentences)
    # if it is with_dropout, transform x
    scaler = None
    if with_dropout:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

    model = full_NN.build_nn_and_train(hidden_layers=hidden_layers, epochs=epochs, x=x, y=y, with_dropout=with_dropout)

    return model, scaler


# model: the trained model used to make the predictions
# first_endings: number_of_samples x 4800, the skip-thought-embedded first endings (used for training the model)
# second_endings: number_of_samples x 4800, the skip-thought-embedded second endings (used for training the model)
# context_sentences: number_of_samples x 1, 4800, the context to be used (used for training the model)
# scaler: the scaler that was used during training (None if no scaling was done)
def predict_right_ending(model, first_endings, second_endings, context_sentences, scaler):

    # get predictions for the validation set
    predicted_right_endings = full_NN.predict(model=model, context=context_sentences, ending1=first_endings,
                                              ending2=second_endings, scaler=scaler)

    return predicted_right_endings


# number_of_splits: determines in how many splits the given data is split
# first_endings: number_of_samples x 4800, the skip-thought-embedded first endings (used for training the model)
# second_endings: number_of_samples x 4800, the skip-thought-embedded second endings (used for training the model)
# right_endings: number_of_samples x 1, 1 if first ending is correct, 2 otherwise (used for training the model)
# context_sentences: number_of_samples x 1, 4800, the context to be used (used for training the model)

# with_scaling: boolean, should be true if data should be normalized before training and false otherwise
# context_sentences: determines implicitly the mode in which the NN is run.
# NC case: number_of_samples x 4800, filled with zeros
# LS case: number_of_samples x 4800, skip-thought embedding of last (4th) context sentence
# FC case: number_of_samples x 4800, encoding of skip-thought embeddings of all 4 context sentences
def do_cross_validation(number_of_splits, first_endings, second_endings, right_endings, context_sentences,
                        hidden_layers, epochs, with_scaling):
    kf = KFold(n_splits=number_of_splits)

    accuracy = []
    for trn_index, val_index in kf.split(first_endings):

        first_endings_trn, first_endings_val = first_endings[trn_index], first_endings[val_index]
        second_endings_trn, second_endings_val = second_endings[trn_index], second_endings[val_index]
        right_endings_trn, right_endings_val = right_endings[trn_index], right_endings[val_index]
        context_sentences_trn, context_sentences_val = context_sentences[trn_index], context_sentences[val_index]

        x, y = create_training_xy(first_endings=first_endings_trn, second_endings=second_endings_trn,
                                  right_endings=right_endings_trn, context_sentences=context_sentences_trn)
        scaler = None
        if with_scaling:
            scaler = StandardScaler()
            scaler.fit(x)
            x = scaler.transform(x)

        model = full_NN.build_nn_and_train(hidden_layers=hidden_layers, epochs=epochs, x=x, y=y, with_dropout=with_scaling)

        predicted_right_endings = full_NN.predict(model=model, context=context_sentences_val, ending1=first_endings_val,
                                                  ending2=second_endings_val, scaler=scaler)

        accuracy.append(accuracy_score(y_true=right_endings_val, y_pred=predicted_right_endings))

    print("accuracies for ", number_of_splits, " splits: ", accuracy)
    print("mean accuracy: ", np.mean(accuracy))
    print("average accuracy: ", np.average(accuracy))

    return accuracy


def create_training_xy(first_endings, second_endings, right_endings, context_sentences=None):
    number_of_samples = len(first_endings)

    # if no context given, initialize context sentences with zero (for NC case)
    if context_sentences is None:
        context_sentences = np.zeros((number_of_samples, 4800))

    x = []
    y = []
    for i in range(number_of_samples):

        # add the last sentence context to each of the endings and append both to training input x
        x.append(first_endings[i] + context_sentences[i])
        x.append(second_endings[i] + context_sentences[i])

        if right_endings[i] == 1:
            # first ending is correct, thus add label 1 for first ending and label 0 for second ending
            y.append(1)
            y.append(0)
        elif right_endings[i] == 2:
            # second ending is correct, thus add label 0 for first ending and label 1 for second ending
            y.append(0)
            y.append(1)
        else:
            raise Exception('runNC: right ending must be 1 or 2, but was: {}'.format(right_endings[i]))

    return x, y


def split_train_and_validation_set(percentage, data):
    number_of_samples = len(data)

    number_of_val_samples = int(percentage * number_of_samples)
    number_of_trn_samples = number_of_samples - number_of_val_samples

    # shuffle the data
    data = shuffle(data, random_state=1)

    # split data into validation and training set
    data_val = data[:number_of_val_samples]
    data_trn = data[number_of_val_samples:]

    return data_trn, data_val


def create_validation_set(percentage, first_endings, second_endings, right_endings, context_sentences=None):
    number_of_samples = len(first_endings)

    number_of_val_samples = int(percentage * number_of_samples)
    number_of_trn_samples = number_of_samples - number_of_val_samples

    # shuffle the data
    first_endings = shuffle(first_endings, random_state=1)
    second_endings = shuffle(second_endings, random_state=1)
    right_endings = shuffle(right_endings, random_state=1)

    # split data into validation and training set
    first_endings_val = first_endings[:number_of_val_samples]
    second_endings_val = second_endings[:number_of_val_samples]
    right_endings_val = right_endings[:number_of_val_samples]

    first_endings_trn = first_endings[number_of_val_samples:]
    second_endings_trn = second_endings[number_of_val_samples:]
    right_endings_trn = right_endings[number_of_val_samples:]

    context_sentences_val = None
    context_sentences_trn = None
    if context_sentences is not None:
        context_sentences = shuffle(context_sentences, random_state=1)
        context_sentences_val = context_sentences[:number_of_val_samples]
        context_sentences_trn = context_sentences[number_of_val_samples:]

    return (first_endings_trn, first_endings_val), (second_endings_trn, second_endings_val), (right_endings_trn, right_endings_val), (context_sentences_trn, context_sentences_val)


