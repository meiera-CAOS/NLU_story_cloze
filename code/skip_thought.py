import sys
sys.path.append('../')
from skipthought import skipthoughts


class SkipThoughtEncoder:
    encoder = None

    def init(self):
        # use skipthought encoding (code in folder skipthought is from https://github.com/ryankiros/skip-thoughts)
        skipthoughts_model = skipthoughts.load_model()
        self.encoder = skipthoughts.Encoder(skipthoughts_model)
