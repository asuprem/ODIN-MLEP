
import os, time, json
from MLEPServer import *




if __name__ == "__main__":
    internalTimer = 0

    # TODO --> sortDataTimes()
    # 
    data = []
    with open('data/data/apriori_to_december_sorted_positive.json','r') as data_file:
        for line in data_file:
            data.append(json.loads(line.strip()))
    
    negatives = []
    with open('data/data/apriori_to_december_negatives.json','r') as data_file:
        for line in data_file:
            negatives.append(json.loads(line.strip()))
    
    trainingData = []
    with open('data/data/aibek_test_converted.json','r') as data_file:
        for line in data_file:
            trainingData.append(json.loads(line.strip()))
    
    # Now we have the data


    MLEPLearner = MLEPLearningServer()
    MLEPPredictor = MLEPPredictionServer()

    #MLEPLearner.train(data=trainingData, models='all')
    #MLEPLearner.addNegatives(negatives)

    # let's do something with it
    for item in data:
        if internalTimer < item['timestamp']:
            internalTimer = long(item['timestamp'])

            # This is the execute loop, but for this implementation. Ideally execute loop is self-sufficient. 
            # But for testing, we ened to manually trigger it
            MLEPLearner.updateTime(internalTimer)
        MLEPPredictor.classify(item)
