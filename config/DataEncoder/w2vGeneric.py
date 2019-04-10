from DataEncoder import DataEncoder

class w2vGeneric(DataEncoder):
    """ Built-in encoder for Generic w2v;"""

    def __init__(self,modelPath = "GoogleNews-vectors-negative300.bin", trainMode = "C", binary=True, unicode_errors = "ignore", limit=None ):
        """
            modelPath -- Name of the w2v model
            trainMode -- ["python","C"]
                            Since this is a gensim wrapper, training mode is required to know how to load the modelfile
            binary -- [True, False]; whether the model is a binary file or non-binary file
            unicode_errors -- ["ignore", "strict"]
            limit -- [None, int]
                        How many words to keep. None means all words are retained

            All options after trainMode are only relevant if trainMode is C

        """
        from gensim.models import KeyedVectors
        from gensim.utils import tokenize
        from numpy import zeros

        self.zeros = zeros
        self.zero_v = self.zeros(shape=(300,))
        self.tokenize = tokenize
        
        self.modelPath = "./config/Sources/" + modelPath
        if trainMode == "C":
            self.model = KeyedVectors.load_word2vec_format(self.modelPath, binary=binary, unicode_errors=unicode_errors, limit=limit)
        else:
            self.model = KeyedVectors.load(self.modelPath)
        
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