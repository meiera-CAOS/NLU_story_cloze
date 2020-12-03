from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

twd = TreebankWordDetokenizer()
input_file_to_postprocess = "../data/ours/generated_sentences_50epochs"
postprocessed_file = input_file_to_postprocess + "_POSTPROCESSED"
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