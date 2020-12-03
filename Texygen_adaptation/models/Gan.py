from abc import abstractmethod

from utils.utils import init_sess


class Gan:
    def __init__(self):
        self.generator = None
        self.discriminator = None
        self.gen_data_loader = None
        self.dis_data_loader = None
        self.sess = init_sess()
        self.epoch = 0
        self.log = None
        self.reward = None

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    def set_data_loader(self, gen_loader, dis_loader):
        self.gen_data_loader = gen_loader
        self.dis_data_loader = dis_loader

    def set_sess(self, sess):
        self.sess = sess

    # ---- gets called ----
    def add_epoch(self):
        self.epoch += 1

    '''
    Each GAN has a list of metrics that you can add. These can calculate some loss.
    This loss gets printed here.
    
    metric.get_score() acts differently per metric:
    while Nll takes the data from our gen_data_loader (filled with data from oracle.txt) to calculate the loss
    the DocEmbSim metric somehow relates the contents of our oracle.txt file and our generator.txt file
    '''

