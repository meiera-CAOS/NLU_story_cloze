# NLU 2nd lecture project in collaboration with Martina, Christina and Virginia

## Requirements
- Python3 (3.7)
- tensorflow (1.13.1)
- Keras-Applications (1.0.7)
- Keras-Preprocessing (1.0.9)
- pandas (0.24.2)
- numpy (1.16.2)
- scipy (1.2.1)
- gensim (3.7.2)
- scikit-learn (0.21.1)
- nltk (3.4.1)
- Theano (1.0.4)

run: ```pip3 install -r requirements.txt ```

## Download data

- place the content of data.zip from this polybox [link](https://polybox.ethz.ch/index.php/s/c3f5dRthmoKVOhK) into nlu_2/data/

- place the following data into nlu_2/Texygen_adaptation/data/

    - word2vec embedding: https://polybox.ethz.ch/index.php/s/mFkjmC9EmPKDzg1
    - training set: https://polybox.ethz.ch/index.php/s/l2wM4RIyI3pD7Tl
    - validation set: https://polybox.ethz.ch/index.php/s/02IVLdBAgVcsJAx

## How to run
First (once) you need to download the NLTK punkt package.
run following commands in the python3 console:
```
>>> import nltk
>>> nltk.download('punkt')
```


### FFNN
To predict which ending fits the story better with the FFNN:

navigate to nlu_2/code/

To get predictions for the test set with our final model:

python3 main_FC_dropout_65_epochs_test.py

Various settings have their respective main files there.

The main_crossvalidation_2016_... files were used to run the experiments with VAL mode of table 1.

The main_no_cross_validate_... files were used to run the LSTM-END mode experiments

process_testreport_all_modes.py was used to produce the results for table 2.

### LSTM
To generate alternate ending sentences to the training data stories with the LSTM:

navigate to nlu_2/lstm/

to run: python3 main.py
### GAN
To generate alternate ending sentences to the training data stories with the GAN:

navigate to nlu_2/Texygen_adaptation/

to run: python3 main.py
### Skip-thought
To compute skip-thought embeddings of sentences:

WARNING: YOU DO NOT WANT TO DO THAT!!!

The precomputed embeddings are in nlu_2/data/ours or [on the polybox](https://polybox.ethz.ch/index.php/s/c3f5dRthmoKVOhK)

They are automatically used by our FFNN.

If you really can't resist...

download the necessary data (several GBs) and place it in nlu_2/data/:
```
wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
```

to compute the embeddings: 

navigate to nlu_2/code/

run the appropriately named ..._create_..._skipthought_embeddings file with the correct arguments,

good luck!

## Declaration of foreign code:
- Skipthought: translated from python2 to python3 and adapted for our purpose.
    - Code: <https://github.com/ryankiros/skip-thoughts>
- Word2Vec: Code and data as provided for project 1.
    - Code: <https://polybox.ethz.ch/index.php/s/890Q7LFpnI6ckqs>
    - Embeddings: <https://polybox.ethz.ch/index.php/s/mFkjmC9EmPKDzg1>
- Texygen: Adapted the textgan model to the STC task.
    - Code: <https://github.com/geek-ai/Texygen>
