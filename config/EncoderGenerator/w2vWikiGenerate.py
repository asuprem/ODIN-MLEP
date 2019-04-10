from EncoderGenerator import EncoderGenerator

class w2vWikiGenerate(EncoderGenerator):
    """ Built-in encoder for Bag of Words.
    
    Current case: Given bow.txt, generate bow.encoder
    
    
    Planned case: EncoderGenerator is given a document. This document's corres[onding bow file is then generated
                    Pipeline in MLEP server includes set of args for a pipeline, including argument for Encoder setup.
                    Encders indexed by name AND args (somehow) (Mongo?) """
    def __init__(self):
        pass


    def generate(self,dimensionSize="5000", seedName="wikipedia"):
        # Check if model already exists
        modelSaveName = "-".join(["w2v","wiki", str(seedName), str(dimensionSize)]) + ".bin"
        modelSavePath = "./config/Sources/"+modelSaveName
        import os
        if os.path.exists(modelSavePath):
            return True
        else:
            # Check if rwa file already exists. If not, create it.
            wikipages = self.getWikipages(dimensionSize, seedName)
            
            import gensim
            from scipy.sparse import csr_matrix
            from gensim.utils import tokenize
            #get tokenized forms
            documents = [gensim.utils.simple_preprocess(item) for item in wikipages]
            model = gensim.models.Word2Vec(documents, size=300, window=10,min_count=2,workers=10)
            model.train(documents, total_examples=len(documents),epochs=10)
            model.save(modelSavePath)
            return True



    def getWikipages(self,dimensionSize, seed):
        import os, pickle       
        
        
        wikipagesFileName = str(seed) + '_' + str(dimensionSize) +'.wikipages'
        wikipagesFilePath = os.path.join('./config/RawSources',wikipagesFileName)
        listOfWikiPages=[]
        listOfWikiTitles={}
        
        if not os.path.exists(wikipagesFilePath):
            import random, wikipedia
            random.seed(a=seed)

            articleTitleNames=[]
            wikiTitlesFileName = 'enwiki-latest-all-titles-in-ns0'
            wikiTitlesFilePath = os.path.join('./config/RawSources',wikiTitlesFileName)
            with open(wikiTitlesFilePath, 'r') as wikiTitlesFile:
                for line in wikiTitlesFile:
                    if line.startswith('!')  or line.startswith('`'):
                        continue
                    articleTitleNames.append(line.strip())
            
            #Now articleTitleNames has list of wikipedia titles. We have to download these, and create td-idf matrix from their texts
            while len(listOfWikiPages) < int(dimensionSize):
                try:
                    _title = random.randint(0, len(articleTitleNames)-1)
                    _title = articleTitleNames[_title].strip()
                    if _title in listOfWikiTitles:
                        continue
                    wiki_text = wikipedia.page(_title)
                    if len(wiki_text.content.split(' ')) < 100:
                            continue
                    listOfWikiTitles[_title] = 1
                    listOfWikiPages.append(wiki_text.content)
                except:
                    continue
            with open(wikipagesFilePath, 'wb') as wikipagesFileName:
                pickle.dump([listOfWikiPages, listOfWikiTitles], wikipagesFileName)
        else:
            with open(wikipagesFilePath, 'rb') as wikipagesFileName:
                wiki_data = pickle.load(wikipagesFileName)
                listOfWikiPages = wiki_data[0]
                listOfWikiTitles = wiki_data[1]
        return listOfWikiPages