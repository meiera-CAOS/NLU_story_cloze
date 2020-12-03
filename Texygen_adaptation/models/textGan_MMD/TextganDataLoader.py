import numpy as np
import pandas as pd

'''
Loads Data for the Generator (LSTM)
'''

class DataLoader():
    def __init__(self, batch_size, seq_length, end_token=0):
        self.batch_size = batch_size
        self.token_stream = []
        self.seq_length = seq_length
        self.end_token = end_token

    # ---- gets called ----
    '''
    - reads data from the data_file (e.g. image/coco.txt) which is stored in oracle.txt in encoded form
    ----> (currently, training data (5 sentences, correct endings)
    - puts it in batches of size batch_size
    - discards batches that are not full
    - stores in self.sequence_batch (which is an array of batches)
    - can then be cycled through by calling next_batch()
    '''
    def create_batches_old(self, data_file):
        self.token_stream = []

        with open(data_file, 'r') as raw:
            for line in raw:
                line = line.strip().split()
                parse_line = [int(x) for x in line]
                if len(parse_line) > self.seq_length:
                    # crop to match the sequence length TODO: change this?
                    self.token_stream.append(parse_line[:self.seq_length])
                else:
                    while len(parse_line) < self.seq_length:
                        parse_line.append(self.end_token)  # fill up / pad with end token
                    if len(parse_line) == self.seq_length:
                        self.token_stream.append(parse_line)  # if sizes match, just append the whole thing

        self.num_batch = int(len(self.token_stream) / self.batch_size)  # find out how many batches we can form
        self.token_stream = self.token_stream[:self.num_batch * self.batch_size]  # discard batches that are not full
        self.sequence_batch = np.split(np.array(self.token_stream), self.num_batch, 0)  # split the array into batches
        self.pointer = 0

    '''
    - reads data from data_file (oracle.txt in our case), each line is a sentence of equal length (filled with padding)
    - expects 5 sentences line after line, belonging to the same story
    - groups them into stories story_i = [[sentence_5*i], [sentence_5*i+1], ..., [sentence_5*i+4]]
    - returns batches of [[sentence_0, sentence_5, ...], [sentence_1, sentence_6, ...],...[sentence_4, sentence_7, ...]]
    (so each batch contains 5 arrays of batch_size, and each of the 5 arrays contains the i-th sentence of these stories
    '''
    def create_batches(self, data_file):
        sentence_1, sentence_2, sentence_3, sentence_4, sentence_5 = [[], [], [], [], []]

        def sentence_array(index):
            i = index % 5
            if i == 0: return sentence_1
            if i == 1: return sentence_2
            if i == 2: return sentence_3
            if i == 3: return sentence_4
            if i == 4: return sentence_5

        count = 0
        with open(data_file, 'r') as sentences:  # each line is a sentence of a story
            for sentence in sentences:
                target_array = sentence_array(count)
                sentence = sentence.strip().split()
                parsed_sentence = [int(x) for x in sentence]  # because our words are encoded as integers
                if len(parsed_sentence) > self.seq_length:
                    # crop to match the sequence length if we have too many words
                    target_array.append(sentence[:self.seq_length])
                else:
                    while len(parsed_sentence) < self.seq_length:
                        parsed_sentence.append(self.end_token)  # pad with end token (0)
                    if len(parsed_sentence) == self.seq_length:
                        target_array.append(parsed_sentence)  # if sizes match, just append the whole thing
                count += 1

        # find out how many batches we can form
        self.num_batch = int(len(sentence_5) / self.batch_size)
        num_elements = self.num_batch * self.batch_size  # number of elements in sub-batches, 5x sub batches per batch.

        # discard the sentences of unfinished batches
        sentence_1 = sentence_1[:num_elements]
        sentence_2 = sentence_2[:num_elements]
        sentence_3 = sentence_3[:num_elements]
        sentence_4 = sentence_4[:num_elements]
        sentence_5 = sentence_5[:num_elements]

        # collect data to format [[all first sentences], ...,  [all fifth sentences]]
        all_data = [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5]
        # (np.array(all_data)).shape is (5 (sub-batches), 64(sentences), 21(tokens))

        # self.sentence_batches = np.array(all_data)

        # line below: splits all_data array into num_batches many arrays with axis 1.
        # returns 64 arrays containing 5 elements each, every array contains the 5, 21 token sentences of same story.
        # is this transformation done with the aim to get batches for the discriminator?
        # using the line below - batch[0] returned an array of shape (5,21) instead of (64,21) expected.

        self.sentence_batches = np.split(np.array(all_data), self.num_batch, 1)
        self.pointer = 0


    # ---- gets called ---- ok
    # cycle through the data (batch by batch), go back to the start if you reached the end
    def next_batch(self):
        ret = self.sentence_batches[self.pointer]  # each batch has 5 arrays of sentence 1 through 5 of our stories
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def reset_pointer(self):
        self.pointer = 0

'''
Loads Data for the Discriminator (CNN)
'''
class DisDataloader():
    def __init__(self, batch_size, seq_length):
        self.batch_size = batch_size
        self.sentences = np.array([])
        self.labels = np.array([])
        self.seq_length = seq_length

    '''
    - reads positive and negative data from files
    - labels positives [1,0] and negatives [0,1]
    - shuffles data and labels the same way
    - puts them into batches of size batch_size
    - discards batches that are not full
    - stores them in self.sentences_batches and self.labels_batches
    - can be cycled through by calling next_batch() (returns tuples of sentence batch and label batch)
    '''
    def load_train_data_old(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    positive_examples.append(parse_line)
        with open(negative_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                if len(parse_line) == self.seq_length:
                    negative_examples.append(parse_line)
        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels))) # shuffle only the indices
        self.sentences = self.sentences[shuffle_indices] # apply that shuffling to the sentences
        self.labels = self.labels[shuffle_indices] # apply that shuffling to the labels too

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)  # find out how many batches we have (rounded down)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]  # discard last batch if not full
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)  # store the batches in arrays
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0


    '''
    - reads positive and negative data from files
    - labels positives [1,0] and negatives [0,1]
    - shuffles data and labels the same way
    - puts them into batches of size batch_size
    - discards batches that are not full
    - stores them in self.sentences_batches and self.labels_batches
    - can be cycled through by calling next_batch() (returns tuples of sentence batch and label batch)
    '''
    def load_train_data(self, positive_file, negative_file):
        # Load data
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            count = 0;
            current_sentence = []
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                current_sentence.extend(parse_line)
                count += 1
                if count == 5:
                    count = 0
                    positive_examples.append(current_sentence)
                    if len(current_sentence) != (5*self.seq_length):
                        print("!!!!!!!!!!!!!!!\tLoading train data, a positive sentence had length " +
                              str(len(current_sentence)) + " instead of the expected length " +
                              str(5 * self.seq_length) + "\t!!!!!!!!!!!!!!!")
                    current_sentence = []

        with open(negative_file)as fin:
            count = 0
            current_sentence = []
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                current_sentence.extend(parse_line)
                count += 1
                if count == 5:
                    count = 0
                    negative_examples.append(current_sentence)
                    if len(current_sentence) != (5*self.seq_length):
                        print("!!!!!!!!!!!!!!!\tLoading train data, a negative sentence had length " +
                              str(len(current_sentence)) + " instead of the expected length " +
                              str(5 * self.seq_length) + "\t!!!!!!!!!!!!!!!")
                    current_sentence = []

        self.sentences = np.array(positive_examples + negative_examples)

        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        self.labels = np.concatenate([positive_labels, negative_labels], 0)

        # Shuffle the data
        shuffle_indices = np.random.permutation(np.arange(len(self.labels))) # shuffle only the indices
        self.sentences = self.sentences[shuffle_indices] # apply that shuffling to the sentences
        self.labels = self.labels[shuffle_indices] # apply that shuffling to the labels too

        # Split batches
        self.num_batch = int(len(self.labels) / self.batch_size)  # find out how many batches we have (rounded down)
        self.sentences = self.sentences[:self.num_batch * self.batch_size]  # discard last batch if not full
        self.labels = self.labels[:self.num_batch * self.batch_size]
        self.sentences_batches = np.split(self.sentences, self.num_batch, 0)  # store the batches in arrays
        self.labels_batches = np.split(self.labels, self.num_batch, 0)

        self.pointer = 0

    # cycle through the batches
    def next_batch(self):
        ret = self.sentences_batches[self.pointer], self.labels_batches[self.pointer]
        self.pointer = (self.pointer + 1) % self.num_batch
        return ret

    def next_batch_including_separated_data(self):
        data, labels = self.next_batch()
        zipped = zip(data, labels)  # [(data, label), ..,(data, label)] where label = [0,1] (pos) or [1,0] (neg)
        positive_data = [val[0] for val in zipped if val[1][0] == 0]  # val[0] = data, val[1][0] == 0 --> label is [0, 1]
        zipped = zip(data, labels)
        negative_data = [val[0] for val in zipped if val[1][1] == 0]  # val[1] = label, val[1][0] == 1 --> label is [1, 0]
        # if one is longer than the other, crop it to the size of the other
        min_length = min(len(positive_data), len(negative_data))
        positive_data = positive_data[:min_length]
        negative_data = negative_data[:min_length]
        return data, labels, positive_data, negative_data


    def reset_pointer(self):
        self.pointer = 0
