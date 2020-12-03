import lstm.load_embedding as load_embedding

import tensorflow as tf

# vocabSize: size of the vocabulary
# embeddingDim: dimension of the embedded word
# returns: returns an embedding, on which a look up can be performed
def createNewWordEmbedding(vocabularySize, embeddingDimension):
    #TODO: do we need xavier_initializer for word_embeddings?
    wordEmbeddings = tf.get_variable("word_embeddings", shape=[vocabularySize, embeddingDimension])
    return wordEmbeddings


def getWord2VecEmbedding(session, vocabularySize, embeddingDimension, vocab):
    embedding = tf.Variable(initial_value=tf.zeros([vocabularySize, embeddingDimension], tf.float32), name='word2vecEmbedding', trainable=True)
    load_embedding.load_embedding(session=session, dim_embedding=embeddingDimension, emb=embedding, path='../data/wordembeddings-dim100.word2vec', vocab=vocab, vocab_size=vocabularySize)
    return embedding