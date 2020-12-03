from time import time

from models.Gan import Gan
from models.textGan_MMD.TextganDataLoader import DataLoader, DisDataloader
from models.textGan_MMD.TextganDiscriminator import Discriminator
from models.textGan_MMD.TextganGenerator import Generator
from utils.utils import *
from utils.text_process import *

from models.textGan_MMD.load_embedding import load_embedding


# ---- gets called ----
class TextganMmd(Gan):
    def __init__(self, oracle=None):
        super().__init__()
        # you can change parameters, generator here
        self.vocab_size = 20  # gets overwritten in init_real_training
        self.emb_dim = 100  # used in LSTM and CNN
        self.hidden_dim = 100  # used in LSTM
        self.sequence_length = 20  # used in LSTM & CNN, represents sentence length, gets changed in init_real_trainng
        self.filter_size = [2, 3]  # used in CNN
        self.num_filters = [100, 200]  # used in CNN
        self.l2_reg_lambda = 0.2  # used in CNN
        self.dropout_keep_prob = 0.75  # not used TODO: remove this?
        self.batch_size = 64
        self.generate_num = 128  # only used in generate_samples to determine the amount of batches together with batch_size
        self.start_token = 0  # used in LSTM

        self.pre_epoch_num = 10  # (original value was 80)
        self.adversarial_epoch_num = 100  # (original value was 100) 400
        self.adversarial_gen_train_steps = 100  # (original value was 100)
        self.adversarial_dis_train_steps = 15  # (original value was 15)

        self.oracle_file = 'save/oracle.txt'
        self.generator_file = 'save/generator.txt'
        self.test_file = 'save/test_file.txt'
        self.validation_file_true = "save/validation_data_true.txt"
        self.validation_file_false = "save/validation_data_false.txt"
        self.GAN_sentences_encoded = "save/GAN_sentences_encoded.txt"
        self.GAN_sentences = "save/GAN_sentences.txt"

    '''
    usually called as generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)
    that means we always per default have get_code = True
    '''
    def generate_samples(self, sess, trainable_model, batch_size, generated_num, output_file):
        generated_samples = []
        for _ in range(int(generated_num / batch_size)):  # loop over all our batches, discard batches that are not full
            # generate_samples should gen 1 sentence per story & return the 4 first sentences + gen
            batch = self.gen_data_loader.next_batch()
            generated_sentences = trainable_model.generate(sess, batch[:4])
            # generated_samples.extend(trainable_model.generate(sess, batch[:4]))

            #reconstruct stories - get sentence 0 - 3 + generated sentence in oder of stories.
            for i in range(batch_size):
                for j in range(5):
                    if j < 4:
                        generated_samples.append(batch[j][i])
                    else:
                        generated_samples.append(generated_sentences[i])

        # write the generated text into the output file (generator_file/generator.txt)
        # TODO: make sure this overwrites the previous generate.txt
        with open(output_file, 'w') as fout:
            for sentence in generated_samples:
                buffer = ' '.join([str(x) for x in sentence]) + '\n'
                fout.write(buffer)
        return

    def train_discriminator(self):
        # IMPORTANT: THIS ASSUMES THERE IS DATA IN THE DIS_DATALOADER -> LOAD DATA BEFORE CALLING THIS!
        # takes data that has been previously loaded into generator.txt (fake endings) and oracle.txt (true endings)
        # iterate through batches by calling next_batch
        for _ in range(3):
            # here the z_h randomness is read from generation of batch
            # x_batch, z_h = self.generator.generate(self.sess, True)

            # temporarily set z_h to a random value independent of our generated data
            z_h = np.random.uniform(low=-.01, high=1, size=[self.batch_size, self.emb_dim])

            # Note: for each generated batch of data, there is a random seed z_h. Optimally, we'd want the z_h we use
            # to be the same for generator and discriminator. Our issue is just that the generator generates several
            # batches of data with a different random seed each time, and we load in the full generated data (several
            # batches) wildly mixed together with ground truth (correct sentences) data. Unless we store the info on
            # which sentence was generated with which seed, we cannot place the same randomness seed in our
            # discriminator... even if we knew it for each sentence, it would be difficult to implement.
            # The code we based this on decided to call generator.generate and combine one batch of freshly generated
            # data with a batch of ground truth data, but even so, this would always make for a perfect 50%50 split of
            # the data, which is not what we want our discriminator to get biased on.
            # The random seed in the generator is always generated with np.random.uniform, so empirically we can say...
            # it might (?) get more robust even, if we use a fresh random seed? Don't quote me on this though.

            sentences, labels, positives, negatives = self.dis_data_loader.next_batch_including_separated_data()

            feed = {
                self.discriminator.input_data: sentences,  # mix of generated and ground truth sentences
                self.discriminator.input_labels: labels,  # labels corresponding to these sentences
                self.discriminator.zh: z_h,  # same randomness as used to init generation...
                self.discriminator.positive_data: positives,  # just used for MMD loss, data that appears in sentences
                self.discriminator.negative_data: negatives,  # and just separated by labels as it's difficult to do so later on
            }

            # discriminator.train_op is d_optimizer.apply_gradients, which returns an "operation" that applies gradients
            # see tensorflow->python->training->optimizer.py
            # the feed just sets values for all placeholder variables in our discriminator
            _ = self.sess.run(self.discriminator.train_op, feed)  # = wildcard... side-effects of run?

    # ---- gets called ----
    def train_generator(self):
        z_h0 = np.random.uniform(low=-.01, high=.01, size=[self.batch_size, self.emb_dim])
        z_c0 = np.zeros(shape=[self.batch_size, self.emb_dim])

        batch = self.gen_data_loader.next_batch()

        for sentence_batch in batch:

            feed = {
                self.generator.h_0: z_h0,  #random init
                self.generator.c_0: z_c0,  #zeroes
                self.generator.y: sentence_batch,  # non generated input sentences
            }

            # run on pretrain loop for the first 4 sentences
            _, prev_state = self.sess.run(fetches=[self.generator.g_updates, self.generator.gen_last_state], feed_dict=feed)
            z_h0 = prev_state[0]
            z_c0 = prev_state[1]

    # ------- gets called ----------
    def init_real_training(self):

        # Loads the ground truth texts and stores them in oracle.txt
        # if load_validation_data, it loads the validation data too and stores them in validation_data_true.txt and
        # in validation_data_false.txt
        # sets sequence length to longest sentence's length in training and testing data (here we only specify training
        # data, as testing data can be left empty)
        # Note: sequence length = the maximum length of the sentences which we wil generate, all shorter will be padded
        # it also sets vocab_size to the longest sentence in the input dataset
        self.sequence_length, self.vocab_size, index_word_dict, self.generate_num, word_index_dict = text_process(load_validation_data=True)

        # Initialize the Discriminator
        #g_embeddings = tf.Variable(tf.random_normal(shape=[self.vocab_size, self.emb_dim], stddev=0.1))
        g_embeddings = tf.Variable(initial_value=tf.zeros([self.vocab_size, self.emb_dim], tf.float32),
                                   name='word2vecEmbedding', trainable=False)
        load_embedding(session=self.sess, dim_embedding=self.emb_dim, emb=g_embeddings,
                       path='data/wordembeddings-dim100.word2vec', vocab=word_index_dict, vocab_size=self.vocab_size)

        print("word embedding loaded")
        discriminator = Discriminator(sequence_length=(self.sequence_length*5), num_classes=2,
                                      emb_dim=self.emb_dim, filter_sizes=self.filter_size, num_filters=self.num_filters,
                                      g_embeddings=g_embeddings,
                                      l2_reg_lambda=self.l2_reg_lambda, dropout_keep_prob=self.dropout_keep_prob)
        self.set_discriminator(discriminator)

        # Initialize the Generator
        generator = Generator(num_vocabulary=self.vocab_size, batch_size=self.batch_size, emb_dim=self.emb_dim,
                              hidden_dim=self.hidden_dim, sequence_length=self.sequence_length,
                              g_embeddings=g_embeddings, discriminator=discriminator, start_token=self.start_token)
        self.set_generator(generator)

        # Initialize the Data Loaders for Generator (gen_dataloader) and Discriminator (dis_dataloader)
        gen_dataloader = DataLoader(batch_size=self.batch_size, seq_length=self.sequence_length)
        dis_dataloader = DisDataloader(batch_size=self.batch_size, seq_length=self.sequence_length)
        self.set_data_loader(gen_loader=gen_dataloader, dis_loader=dis_dataloader)
        return index_word_dict


    def train_real(self):
        from utils.text_process import code_to_text
        from utils.text_process import tokenize_file
        # create dictionaries, wi_dict word to index mapping , iw_dict: indey to word mapping
        iw_dict = self.init_real_training()

        def generate_output_sentences(sess, trainable_model, batch_size, generated_num):
            generated_sentences = []
            
            for _ in range(int(
                    generated_num / batch_size)):  # loop over all our batches, discard batches that are not full
                # generate_samples should gen 1 sentence per story & return the 4 first sentences + gen
                batch = self.gen_data_loader.next_batch()

                gen_samples = trainable_model.generate(sess, batch[:4])

                if np.array(gen_samples).shape != np.array(batch[0]).shape:
                    raise Exception('expected format of gen_samples is one batch with '
                                    'dimensions batch_size x sequence_length '.format(gen_samples))

                generated_sentences.extend(gen_samples)

            with open(self.GAN_sentences_encoded, 'w') as fout:
                for poem in generated_sentences:
                    buffer = ' '.join([str(x) for x in poem]) + '\n'
                    fout.write(buffer)

            with open(self.GAN_sentences_encoded, 'r') as file:
                # read the encoded words (encoded as their id in the vocab)
                codes = tokenize_file(self.GAN_sentences_encoded)
            with open(self.GAN_sentences, 'w') as outfile:
                # decode the words into sentences and write them into the test file
                outfile.write(code_to_text(codes=codes, dictionary=iw_dict))
            return

        # initialize all trainable variables
        print("Initializing Network...")
        self.sess.run(tf.global_variables_initializer())
        self.log = open('experiment-log-textgan-real.csv', 'w')

        print("Creating batches...")
        # loads data from oracle.txt into the (generator's) data loader
        # and batches it so that it can be cycled through with next_batch()
        self.gen_data_loader.create_batches(self.oracle_file)

        print("Generating Samples...")
        # writes generated samples to generator.txt, first time generating sentences.
        self.generate_samples(self.sess, self.generator, self.batch_size, self.generate_num, self.generator_file)

        print('start pre-train generator:')
        for epoch in range(self.pre_epoch_num):
            start = time()

            # pre trained loss for one epoch using MLE
            # calls generator.pretrain_step(sess, batch) with batches of data from the generator's data loader
            # as specified a few lines above, this data actually comes from oracle.txt
            loss = pre_train_epoch(self.sess, self.generator, self.gen_data_loader, self.batch_size, self.emb_dim)

            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start) + '\t loss:' + str(loss))
            self.add_epoch()  # note self.epoch != epoch loop variable

            generate_output_sentences(self.sess, self.generator, self.batch_size, self.generate_num)
            post_process(self.GAN_sentences, self.epoch)

        # load data from validation set into dis_data_loader
        self.dis_data_loader.load_train_data(positive_file=self.validation_file_true, negative_file=self.validation_file_false)
        print('start pre-train discriminator:')
        for epoch in range(self.pre_epoch_num):
            print('epoch:' + str(epoch))
            self.train_discriminator()  # uses the data in dis_data_loader

        print('adversarial training:')
        for epoch in range(self.adversarial_epoch_num):
            print('epoch:' + str(epoch))
            start = time()
            # training steps of generator
            for index in range(self.adversarial_gen_train_steps):
                self.train_generator()
            end = time()
            print('epoch:' + str(self.epoch) + '\t time:' + str(end - start))
            self.add_epoch()

            # training steps of discriminator
            # load data from generator.txt and oracle.txt into dis_data_loader
            self.dis_data_loader.load_train_data(positive_file=self.oracle_file, negative_file=self.generator_file)
            for _ in range(self.adversarial_dis_train_steps):
                self.train_discriminator()

            generate_output_sentences(self.sess, self.generator, self.batch_size, self.generate_num)
            post_process(self.GAN_sentences, self.epoch)

        # Generates final sentences to GAN_sentences.txt
        generate_output_sentences(self.sess, self.generator, self.batch_size, self.generate_num)
        post_process(self.GAN_sentences)
