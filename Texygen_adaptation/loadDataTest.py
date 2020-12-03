from utils.text_process import *

batch = [[0], [1], [2], [3], [4]]
print(batch[0])


#To run change hardcoded datalocation in data_loader to ../../data...
_, _, _, train_stories = load_training_data()
# [:2] to only check for 2 stories.
two_stories = train_stories[:2]
# print(two_stories)

tokenized = get_tokenized(two_stories)
counter = 0
for story in tokenized:
    print("Tokenized story ID ", counter, ": ", story)
    counter += 1

# does empty list impact?
word_set = get_word_list(tokenized + list())
[word_index_dict, index_word_dict] = get_dict(word_set)

seq_len = max_seq_length(tokenized)
code = text_to_code(tokenized, word_index_dict, seq_len)
text = code_to_text(code, index_word_dict)

# assert decode, encode is same as before.

# max sentence length of all train set stories from our original dataset is 21.
# print(max_seq_length(tokenized))
