#import nltk
#nltk.download('punkt')

from models.textGan_MMD.Textgan import TextganMmd

if __name__ == '__main__':
    gan = TextganMmd()
    gan.train_real()
