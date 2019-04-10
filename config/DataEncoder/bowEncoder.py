from DataEncoder import DataEncoder

class bowEncoder(DataEncoder):
    """ Built-in encoder for bag of words; """

    def __init__(self,modelFileName="bow.model"):
        from sklearn.externals import joblib
        from sklearn.feature_extraction.text import CountVectorizer

        modelFilePath = "config/Sources/" + modelFileName
        self.model = joblib.load(modelFilePath)
        
    def encode(self, data):
        """ data MUST be a list of string """
        try:
            return self.model.transform(data)
        except ValueError:
            return self.model.transform([data])

    def batchEncode(self, data):
        """ batch encode. data must be a list of stringds"""
        return self.model.transform(data)