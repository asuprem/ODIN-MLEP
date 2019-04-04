from DataEncoder import DataEncoder

class w2vGoogleNews(DataEncoder):
    """ Built-in encoder for Google w2v; limited to 100K most common words """

    def __init__(self,):
        from gensim.models import KeyedVectors
        from gensim.utils import tokenize
        from numpy import zeros
        self.model = KeyedVectors.load_word2vec_format('config/Sources/GoogleNews-vectors-negative300.bin', binary=True, unicode_errors='ignore', limit=100000)
        self.zeros = zeros
        self.zero_v = self.zeros(shape=(300,))
        self.tokenize = tokenize
    def encode(self, data):
        """ data MUST be a string """
        tokens = list(self.tokenize(data))
        # this is for possibly empty tokens
        transformed_data = self.zeros(shape=(300,))
        if not tokens:
            pass
        else:
            for word in tokens:
                transformed_data += self.model[word] if word in self.model else self.zero_v
            transformed_data/=len(tokens)
        return transformed_data



    def batchEncode(self, data):
        """ batch encode. data must be a list of stringds"""
        max_len = len(data)
        transformed_data = self.zeros(shape=(max_len,300))
        
        for idx, sentence in enumerate(data):
            transformed_data[idx] = self.encode(sentence)
        return transformed_data