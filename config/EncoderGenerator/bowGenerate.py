from EncoderGenerator import EncoderGenerator

class bowGenerate(EncoderGenerator):
    """ Built-in encoder for Bag of Words.
    
    Current case: Given bow.txt, generate bow.encoder
    
    
    Planned case: EncoderGenerator is given a document. This document's corres[onding bow file is then generated
                    Pipeline in MLEP server includes set of args for a pipeline, including argument for Encoder setup.
                    Encders indexed by name AND args (somehow) (Mongo?) """
    def __init__(self):
        pass

    def generate(self,rawFileName="bow.txt", modelFileName="bow.model"):

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