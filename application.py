import os, time, json
from MLEPServer import *




if __name__ == "__main__":
    internalTimer = 0

    # TODO --> sortDataTimes()
    # 
    data = []
    with open('data/data/2014_to_dec2018.json','r') as data_file:
        for line in data_file:
            data.append(json.loads(line.strip()))
    
    negatives = []
    with open('data/data/collectedIrrelevant.json','r') as data_file:
        for line in data_file:
            negatives.append(json.loads(line.strip()))
    
    trainingData = []
    with open('data/data/initialTrainingData.json','r') as data_file:
        for line in data_file:
            trainingData.append(json.loads(line.strip()))
    
    # Now we have the data


    MLEPLearner = MLEPLearningServer()
    MLEPPredictor = MLEPPredictionServer()

    # Train with raw training data (for now)
    # Assumptions - there is a 'text' field; assume we have access to a w2v encoder
    MLEPLearner.initialTrain(traindata=trainingData)
    std_flush("Completed training at", readable_time())
    MLEPLearner.addNegatives(negatives)

    # let's do something with it
    totalCounter = []
    mistakes = []
    for item in data:
        if internalTimer < item['timestamp']:
            internalTimer = long(item['timestamp'])

            # This is the execute loop, but for this implementation. Ideally execute loop is self-sufficient. 
            # But for testing, we ened to manually trigger it
            MLEPLearner.updateTime(internalTimer)
        classification = MLEPPredictor.classify(item, MLEPLearner)
        totalCounter.append(1)
        if classification != item['label']:
            mistakes.append(1.0)
        else:
            mistakes.append(0.0)
        if len(totalCounter) % 100 == 0 and len(totalCounter)>0:
            std_flush("Completed", len(totalCounter), " samples, with running error (past 100) of", sum(mistakes[-100:])/sum(totalCounter[-100:]))
        
