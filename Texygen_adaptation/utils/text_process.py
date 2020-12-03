# coding=utf-8
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


def text_to_code(tokens, dictionary, seq_len):
    code_str = ""
    eos_code = 0
    for story in tokens:
        for sentence in story:
            index = 0
            for word in sentence:
                code_str += (str(dictionary[word]) + ' ')
                index += 1
            while index < seq_len:
                code_str += (str(eos_code) + ' ')
                index += 1
            code_str += '\n'
    return code_str


# ------- gets called ----------
def code_to_text(codes, dictionary):
    text = ""
    eos_code = "0"
    eos_token = "."

    for line in codes:
        index = 0
        length = len(line) - 1
        for number in line:
            token = dictionary[str(number)]

            # if we see a '.', add it to the text but stop parsing this line
            # if we see an EOS, replace it by a '.' and stop parsing this line
            # if we arrive at the end of sentence, replace the last token with a '.'
            if token == eos_token or number == eos_code or index == length:
                text += "."
                break

            # otherwise just add the token and a space
            text += (token + ' ')
            index += 1
        text += '\n'
    return text


def get_tokenized(stories):
    tokenized_stories = list()
    for story in stories:
        tokenized_story = list()
        for sentence in story:
            sentence = nltk.word_tokenize(sentence.lower())
            tokenized_story.append(sentence)
        tokenized_stories.append(tokenized_story)
    return tokenized_stories
# tokenized is a list of lists a_i, where a_i is the i-th sentence tokenized,
# eg: tokenized for 2 stories:


''' Tokenized story ID 0: [['kelly', 'found', 'her', ..., 'of', 'memories', '.'],
   ...
    ['kelly', 'successfully', 'made', 'a', 'pizza', 'from', 'her', 'grandmother', "'s", 'recipe', '.']]
    Tokenized story ID 1: [['leo', 'wore', 'a', 'toupee', 'and', 'was', 'anxious', 'about', 'it', '.'],
    ...
    ['his', 'dog', 'leaped', 'and', 'caught', 'it', ',', 'and', 'he', 'quickly', 'ran', 'home', '.']]'''


def tokenize_file(file):
    tokenized = list()

    with open(file) as raw:
        # text in raw represents one line of text, hence 1 sentence in original data format
        for text in raw:
            text = nltk.word_tokenize(text.lower())
            tokenized.append(text)
    # tokenized is a list of lists a_i, where a_i is the i-th sentence tokenized,
    # eg: tokenized: [['a', 'brown', 'and', 'white', 'dog', 'lays', 'on', 'a', 'bed'],
    # ['several', 'people', 'skiing', 'on', 'a', 'trail', 'at', 'night', '.']]
    return tokenized


def get_word_list(tokens):
    word_set = list()
    for story in tokens:
        for sentence in story:
            for word in sentence:
                word_set.append(word)
    return list(set(word_set))


def get_dict(word_set):
    word_index_dict = dict()
    index_word_dict = dict()

    index = 0
    for word in word_set:
        word_index_dict[word] = str(index)
        index_word_dict[str(index)] = word
        index += 1
    return word_index_dict, index_word_dict


def text_process(load_validation_data=False):
    # get_tokenized now returns a list of list: sentences per stories - each sentence is tokenized.
    _, _, _, train_stories, num_stories = load_training_data()
    train_tokens = get_tokenized(train_stories)

    validation_tokens_true = list()
    validation_tokens_false = list()

    if load_validation_data:
        validation_tokens_true, validation_tokens_false = load_tokenized_validation_data()

    word_set = get_word_list(train_tokens + validation_tokens_true + validation_tokens_false)
    word_set.insert(0, "EOS")

    [word_index_dict, index_word_dict] = get_dict(word_set)
    sequence_len = 21  # set 0 and uncomment below to calculate, 21 is for train_stories.csv, 19 for validation data

    '''
    if not load_validation_data:
        max_seq_length(train_tokens)
    else:
        max_seq_validation = max(max_seq_length(validation_tokens_true), max_seq_length(validation_tokens_false))
        sequence_len = max(max_seq_length(train_tokens), max_seq_validation)
    '''

    # turn the input data into code by replacing each word by its index in the dictionary
    # then write it to our oracle.txt
    # Note: oracle.txt seems to get overwritten later on, because it contains very low integers in the end,
    # even though our vocabulary is usually much larger. Still need to investigate where it changes
    with open('save/oracle.txt', 'w') as outfile:
        outfile.write(text_to_code(train_tokens, word_index_dict, sequence_len))

    if load_validation_data:
        # writes code of test tokens into eval_data.txt
        with open('save/validation_data_true.txt', 'w') as outfile:
            outfile.write(text_to_code(validation_tokens_true, word_index_dict, sequence_len))

        with open('save/validation_data_false.txt', 'w') as outfile:
            outfile.write(text_to_code(validation_tokens_false, word_index_dict, sequence_len))

    return sequence_len, len(word_index_dict), index_word_dict, num_stories, word_index_dict


def max_seq_length(tokens):
    seq_len = 0
    for story in tokens:
        longest_sentence = len(max(story, key=len))
        if longest_sentence > seq_len:
            seq_len = longest_sentence
    return seq_len


def load_validation_data():
    # load validation data
    val_data = pd.read_csv("data/cloze_test_val__spring2016 - cloze_test_ALL_val.csv")
    print("Labels: ", val_data.columns.tolist())
    val_right_ending_nr = val_data[['AnswerRightEnding']].values
    val_context_sentences = val_data.iloc[:, 1:5].values
    val_ending_sentence1 = val_data[['RandomFifthSentenceQuiz1']].values
    val_ending_sentence2 = val_data[['RandomFifthSentenceQuiz2']].values
    return val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2


def load_tokenized_validation_data():
    val_right_ending_nr, val_context_sentences, val_ending_sentence1, val_ending_sentence2 = load_validation_data()
    ending_numbers = [x for [x] in val_right_ending_nr]  # gives us an array of [1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 2, ...]
    endings1 = [x for [x] in val_ending_sentence1]
    endings2 = [x for [x] in val_ending_sentence2]

    true_stories = val_context_sentences.tolist()
    false_stories = val_context_sentences.tolist()

    num_stories = len(ending_numbers)
    for i in range(num_stories):
        if ending_numbers[i] == 1:
            true_stories[i].append(endings1[i])
            false_stories[i].append(endings2[i])
        else:
            true_stories[i].append(endings2[i])
            false_stories[i].append(endings1[i])

    true_tokenized = get_tokenized(true_stories)
    false_tokenized = get_tokenized(false_stories)
    return true_tokenized, false_tokenized


def load_training_data():
    # load training data
    train_data = pd.read_csv("data/train_stories.csv")
    print("Training data: ", train_data.head())
    print("Labels: ", train_data.columns.tolist())
    train_context_sentences = train_data.iloc[:, 2:6].values
    train_ending_sentence = train_data[['sentence5']].values
    train_story_title = train_data[['storytitle']].values
    train_sentences = train_data.iloc[:, 2:7].values
    num_stories = len(train_sentences)
    return train_context_sentences, train_ending_sentence, train_story_title, train_sentences, num_stories


def post_process(file, index=None):
    twd = TreebankWordDetokenizer()
    input_file_to_postprocess = file

    if index is not None:
        postprocessed_file = file[:-4] + str(index) + "_POSTPROCESSED"
    else:
        postprocessed_file = file[:-4] + "_POSTPROCESSED"
    sentences = open(input_file_to_postprocess, "r")  # opens file for reading
    file = open(postprocessed_file, "w")

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tokenized = " ".join(tokens)
        toks = word_tokenize(tokenized)
        detokenized = twd.detokenize(toks)
        capitalized = detokenized.capitalize()
        file.write(capitalized)
        file.write("\n")
    file.close()
