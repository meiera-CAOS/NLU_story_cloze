import tensorflow as tf



class Discriminator(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes,
            emb_dim, filter_sizes, num_filters, g_embeddings=None,
            l2_reg_lambda=0.0, dropout_keep_prob=1):
        self.embbeding_mat = g_embeddings

        print("Initializing Discriminator with sequence length " + str(sequence_length))

        # Placeholders for input, output and dropout
        # Note: first dimension None means that it's variable length, and we can put in batches of arbitrary batch size
        self.input_data = tf.placeholder(tf.int32, [None, sequence_length], name="input_data")
        self.input_labels = tf.placeholder(tf.float32, [None, num_classes], name="input_labels")

        # positive and negative data inputted shall be the same as in input_data, just separated according to their
        # labels. we do the filtering outside, because it's easier to filter arrays than tensors.
        # these two values are just used to calculate the MMD loss. Note: if input_data has no exact 50%50 balance of
        # positive and negative data (real vs fake sentences), we crop whichever of positive_data or negative_data is
        # longer, since they need to be equal length for MMD loss to work. we assume that the resulting loss is not
        # far off from the actual MMD loss.
        self.positive_data = tf.placeholder(tf.int32, [None, sequence_length], name="positive_data")
        self.negative_data = tf.placeholder(tf.int32, [None, sequence_length], name="negative_data")

        # random seed
        self.zh = tf.placeholder(tf.float32, [None, emb_dim], name="zh")

        # dropout_keep_prob: 1 = drop none, 0 = drop all (suggested: 0.5 for training, 1 for prediction)
        self.dropout_keep_prob = dropout_keep_prob  # probability of keeping a neuron in the dropout layer
        self.filter_sizes = filter_sizes  # the amount of words we want our convolutional filters to cover, e.g. [3,4,5]
        self.num_filters = num_filters  # number of filters per filter size (len(filter_sizes) * num_filters in total)
        self.sequence_length = sequence_length  # length of our sentences times 5 (as we have 5 padded sentences)
        self.num_classes = num_classes  # classification classes, in our case 2, (real/fake)

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        with tf.variable_scope('discriminator'):
            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):  # embedding does not have GPU support
                self.W = tf.Variable(  # embedding matrix that we learn during training
                    tf.random_uniform([emb_dim, emb_dim], -1.0, 1.0),  # TODO: in the tutorial I found, they use [vocab_size, emb_dim], is this right?
                    name="W")

            # here we start building our convolutional layers
            self.W_conv = list()  # in feature(...) we pool over these
            self.b_conv = list()
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, emb_dim, 1, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    self.W_conv.append(W)
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    self.b_conv.append(b)

            num_filters_total = sum(self.num_filters)  # TODO: in the tutorial I found they use num_filters * len(filter_sizes)
            with tf.name_scope("output"):
                # Wo, bo used for loss in predict function
                self.Wo = tf.Variable(tf.truncated_normal([num_filters_total, self.num_classes], stddev=0.1), name="W")
                self.bo = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")

            # recon layer
            with tf.name_scope("recon"):
                self.Wzh = tf.Variable(tf.truncated_normal([num_filters_total, 1], stddev=0.1), name="Wz")
                self.bzh = tf.Variable(tf.constant(0.0, shape=[1]), name="bz")

            input_data_embedded = tf.nn.embedding_lookup(self.embbeding_mat, self.input_data)  # batch_size x seq_length x g_emb_dim
            scores, _, _ = self.predict(input_x=input_data_embedded)

            def compute_pairwise_distances(x, y):
                """Computes the squared pairwise Euclidean distances between x and y.
                Args:
                  x: a tensor of shape [num_x_samples, num_features]
                  y: a tensor of shape [num_y_samples, num_features]
                Returns:
                  a distance matrix of dimensions [num_x_samples, num_y_samples].
                Raises:
                  ValueError: if the inputs do no matched the specified dimensions.
                """

                if not len(x.get_shape()) == len(y.get_shape()) == 2:
                    raise ValueError('Both inputs should be matrices.')

                if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
                    raise ValueError('The number of features should be the same.')

                norm = lambda x: tf.reduce_sum(tf.square(x), 1)

                # By making the `inner' dimensions of the two matrices equal to 1 using
                # broadcasting then we are essentially substracting every pair of rows
                # of x and y.
                # x will be num_samples x num_features x 1,
                # and y will be 1 x num_features x num_samples (after broadcasting).
                # After the substraction we will get a
                # num_x_samples x num_features x num_y_samples matrix.
                # The resulting dist will be of shape num_y_samples x num_x_samples.
                # and thus we need to transpose it again.
                return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

            def gaussian_kernel_matrix(x, y, sigmas=None):
                r"""Computes a Guassian Radial Basis Kernel between the samples of x and y.
                We create a sum of multiple gaussian kernels each having a width sigma_i.
                Args:
                  x: a tensor of shape [num_samples, num_features]
                  y: a tensor of shape [num_samples, num_features]
                  sigmas: a tensor of floats which denote the widths of each of the
                    gaussians in the kernel.
                Returns:
                  A tensor of shape [num_samples{x}, num_samples{y}] with the RBF kernel.
                """
                if sigmas is None:
                    sigmas = [
                        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                        1e3, 1e4, 1e5, 1e6
                    ]
                beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))

                dist = compute_pairwise_distances(x, y)

                s = tf.matmul(beta, tf.reshape(dist, (1, -1)))

                return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

            def calc_mmd(x, y):
                cost = tf.reduce_mean(gaussian_kernel_matrix(x, x))
                cost += tf.reduce_mean(gaussian_kernel_matrix(y, y))
                cost -= 2 * tf.reduce_mean(gaussian_kernel_matrix(x, y))

                # We do not allow the loss to become negative.
                cost = tf.where(cost > 0, cost, 0, name='value')

                return cost

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                batch_size = tf.shape(scores)[0]
                # 2-wide, batchsize high score/prediction array (percentages of true/fake)
                pos_score = tf.slice(scores, begin=[0, 0], size=[batch_size, 1])
                # 2-wide, batchsize high label array ([1,0] or [0,1] for true/fake)
                pos_label = tf.slice(self.input_labels, begin=[0, 0], size=[batch_size, 1])
                # distance between the predictions and the labels (norm of order 1 gives a 1-wide batchsize high array)
                gan_loss = tf.log(tf.norm(pos_score - pos_label, ord=1))
                print("gan loss")
                print(gan_loss)

                # separate input x and y according to labels, x and y need to be the same size, but as we
                # cannot guarantee the same size (we take randomly selected and mixed batches), we discard data of
                # ground truths or fake generated sentences, whichever set has more sentences. This gives us a small
                # divergence from the actual mmd loss but we can live with that. (Especially because our approach
                # removes the training bias that all data batches are an exact 50%50 split between true and fake data.)
                x_feature = self.feature(input_x=self.positive_data, name='x (positives)')
                y_feature = self.feature(input_x=self.negative_data, name='y (negatives)')
                mmd_loss = calc_mmd(x_feature, y_feature)
                print("mmd loss")
                print(mmd_loss)

                # z_hat = tf.matmul(x_feature, self.Wzh)
                # recon_loss = - tf.square(tf.norm(tf.subtract(z_hat, self.zh), axis=1))  # TODO: fix error here...
                self.loss = tf.reduce_mean(gan_loss) + l2_reg_lambda * l2_loss + 0.1 * mmd_loss  # + 0.1 * recon_loss

        self.params = [param for param in tf.trainable_variables() if 'discriminator' in param.name]
        d_optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = d_optimizer.compute_gradients(self.loss, self.params, aggregation_method=2)
        self.train_op = d_optimizer.apply_gradients(grads_and_vars)

    def feature(self, input_x, name = '', custom_sequence_length=None):
        if not custom_sequence_length:
            custom_sequence_length = self.sequence_length
        if len(input_x.get_shape()) == 2:
            # in case input_x : batch_size x seq_length [tokens] --> not yet embedded, we have to embed it
            input_x = tf.nn.embedding_lookup(self.embbeding_mat, input_x)
        # input_x:  batch_size x seq_length x g_emb_dim
        pooled_outputs = []
        index = -1
        embedded_chars = tf.scan(lambda a, x: tf.matmul(x, self.W), input_x)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)  # adds a 4th dimension, needed for tf.nn.conv2d
        for filter_size, _ in zip(self.filter_sizes, self.num_filters):
            index += 1
            with tf.name_scope("conv-maxpool-%s-midterm" % filter_size):
                # Convolution Layer
                conv = tf.nn.conv2d(
                    embedded_chars_expanded,
                    self.W_conv[index],  # pool over the embedding matrix W
                    strides=[1, 1, 1, 1],
                    padding="VALID",  # slide filter over sentences without padding edges -> narrow convolution
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, self.b_conv[index]), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,  # pool over b (h is defined as a relu depending on b above)
                    ksize=[1, custom_sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',  # slide filter over sentences without padding edges -> narrow convolution
                    name="pool")
                pooled_outputs.append(pooled)  # collect the output of all filter sizes

        # Combine all the pooled features
        num_filters_total = sum(self.num_filters)  # TODO: here again, the tutorial uses num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)  # concatenate the outputs of all filter sizes
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])  # flatten to size [batch_size, num_filters_total]
        return h_pool_flat

    def predict(self, input_x):
        # input_x:  batch_size x seq_length (17 for coco mini data) x g_emb_dim (32)
        l2_loss = tf.constant(0.0)
        d_feature = self.feature(input_x)
        # Add dropout
        with tf.name_scope("dropout"):  # dropout: suggested 0.5 for training, 1.0 (no dropout) during prediction
            h_drop = tf.nn.dropout(d_feature, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            l2_loss += tf.nn.l2_loss(self.Wo)  # TODO: l2_loss never used. Also, does tf.nn.l2_loss have side effects?
            l2_loss += tf.nn.l2_loss(self.bo)
            scores = tf.nn.xw_plus_b(h_drop, self.Wo, self.bo, name="scores")  # actual predicted values
            ypred_for_auc = tf.nn.softmax(scores)
            predictions = tf.argmax(scores, 1, name="predictions")  # predicted values with softmax applied

        return scores, ypred_for_auc, predictions
