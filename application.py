# pylint: disable=no-value-for-parameter
import os, time, json, sys, pdb
from MLEPServer import MLEPLearningServer, MLEPPredictionServer
from utils import std_flush, readable_time, load_json
import click

from config.DataModel.BatchedLocal import BatchedLocal
from config.DataSet.PseudoJsonTweets import PseudoJsonTweets

"""
Arguments

python application.py experimentName [updateSchedule] [weightMethod] [selectMethod] [filterMethod] [kVal]

"""


@click.command()
@click.argument('experimentname')
@click.option('--update', type=int)
@click.option('--weights', type=click.Choice(["unweighted", "performance"]))
@click.option('--select', type=click.Choice(["train", "historical", "historical-new", "historical-updates","recent","recent-new","recent-updates"]))
@click.option('--filter', type=click.Choice(["no-filter", "top-k", "nearest"]))
@click.option('--kval', type=int)
def main(experimentname, update, weights, select, filter, kval):
    # We'll load thhe config file, make changes, and write a secondary file for experiments
    mlepConfig = load_json('./config/configuration/MLEPServer.json')
    if update is not None:
        mlepConfig["config"]['update_schedule'] = update
    if weights is not None:
        mlepConfig["config"]['weight_method'] = weights
    if select is not None:
        mlepConfig["config"]['select_method'] = select
    if filter is not None:
        mlepConfig["config"]['filter_select'] = filter
    if kval is not None:
        mlepConfig["config"]['k-val'] = kval
    
    PATH_TO_CONFIG_FILE = './config/configuration/ExperimentalConfig.json'
    with open(PATH_TO_CONFIG_FILE, 'w') as write_:
        write_.write(json.dumps(mlepConfig))

    
    # Where to save data:
    #pdb.set_trace()
    writePath = 'dataCollect.csv'
    savePath = open(writePath, 'a')
    savePath.write(experimentname + ',')
    
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
    
    trainingData = BatchedLocal(data_source='data/data/initialTrainingData.json', data_mode="collected", data_set_class=PseudoJsonTweets)

    """
    BatchedLocal.getData() --> return list of DataSet objects
    BatchedLocal.getLabels()

    """
    

    # Let's consider three data delivery models:
    #   - Batched
    #   - Streaming (single example)
    #   - Streaming batched (multiple examples at once)

    # We also have multiple data models - how a single piece of data is delivered:
    #   - TextPseudoJsonModel 
    #   - ImageModel
    #   - VideoModel
    #   - NumericModel
    # Each model has these required methods
    #   
    
    # Now we have the data
    MLEPLearner = MLEPLearningServer(PATH_TO_CONFIG_FILE)
    MLEPPredictor = MLEPPredictionServer()

    # Train with raw training data (for now)
    # Assumptions - there is a 'text' field; assume we have access to a w2v encoder

    # We'll pass a training data model...
    # datamodel is a streaming data model??? --> look at streaming in sci-kit multiflow

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
        if len(totalCounter) % 1000 == 0 and len(totalCounter)>0:
            std_flush("Completed", len(totalCounter), " samples, with running error (past 100) of", sum(mistakes[-100:])/sum(totalCounter[-100:]))
        if len(totalCounter) % 100 == 0 and len(totalCounter)>0:
            savePath.write(str(sum(mistakes[-100:])/sum(totalCounter[-100:]))+',')
        # Perform data collection???
    savePath.write('\n')
    savePath.close()    

    MLEPLearner.shutdown()

    std_flush("\n-----------------------------\nCOMPLETED\n-----------------------------\n")


if __name__ == "__main__":
    main()