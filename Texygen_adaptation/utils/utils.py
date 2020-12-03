import numpy as np
import tensorflow as tf

'''
def generate_samples(sess, trainable_model, batch_size, generated_num, output_file=None, get_code=True):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    codes = list()
    if output_file is not None:
        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x) for x in poem]) + '\n'
                fout.write(buffer)
                if get_code:
                    codes.append(poem)
        return np.array(codes)
    codes = ""
    for poem in generated_samples:
        buffer = ' '.join([str(x) for x in poem]) + '\n'
        codes += buffer
    return codes
'''


def init_sess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    return sess


# ---- gets called ----
def pre_train_epoch(sess, trainable_model, data_loader, batch_size, emb_dim):
    # Pre-train the generator using MLE for one (or more?) epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()

        hidden_state = np.zeros(shape=[batch_size, emb_dim])
        cell_state = np.zeros(shape=[batch_size, emb_dim])

        for i in range(5):
            g_loss, last_state = trainable_model.pretrain_step(sess, batch[i], hidden_state, cell_state)
            supervised_g_losses.append(g_loss)
            hidden_state = last_state[0]
            cell_state = last_state[1]

    return np.mean(supervised_g_losses)
