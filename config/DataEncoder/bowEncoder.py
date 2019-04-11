from DataEncoder import DataEncoder

class bowEncoder(DataEncoder):
    """ Built-in encoder for bag of words; """

    def __init__(self,):
        pass

    def setup(self, modelFileName="bow.model"):
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

    def failCondition(self,rawFileName="bow.txt", modelFileName="bow.model"):
        
        bowFilePath = "config/RawSources/" + rawFileName

        from sklearn.feature_extraction.text import CountVectorizer
        with open(bowFilePath, 'r') as bow_file:
            bow_doc = bow_file.read()
        
        #Create the vectorizer and fit the bow document to create the vocabulary
        vectorizer = CountVectorizer()
        vectorizer.fit([bow_doc])
        
        # Save the Encoder
        from sklearn.externals import joblib
        joblib.dump(vectorizer, "config/Sources/"+modelFileName)
        return True        

        #self.transformed_data = vectorizer.transform(self.source_data['text'].values)