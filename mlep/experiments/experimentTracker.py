import os, time, json, sys, pdb, click
import utils

import MLEPServer
from config.DataModel import BatchedLocal
from config.DataModel import StreamLocal
from config.DataSet import PseudoJsonTweets

import mlflow

from sklearn.model_selection import ParameterGrid

def main():
    # We are tracking drift adaptivity
    # namely labeled drift detection, unlabeled-ensembleagreement versus some scheduled runs for now
    # So we are running three experiments

    # set up scheduled params:
    scheduled_param_grid = {  "update": [("2592000000", "M"), ( "1210000000","F")], 
                    "weights":[( "unweighted","U"),( "performance","P" )], 
                    "select":[( "train","TT" ) , (  "recent","RR" ) , ( "recent-new","RN" ) , ( "recent-updates","RU" ) , ( "historical-new","HN" ) , ( "historical-updates","HU" ) , ( "historical","HH" )],
                    "filter":[("no-filter","F") , ("top-k","T"),("nearest","N")],
                    "kval":[("5","5")],
                    "allow_update_schedule": [True]}
    pdb.set_trace()
    scheduled_param = ParameterGrid(scheduled_param_grid)

    # Set up explicit drift detection params
    explicit_drift_param_grid = {   "allow_explicit_drift": [True],
                                    "explicit_drift_class": ["LabeledDriftDetector"],
                                    "explicit_drift_mode":["DDM", "EDDM", "ADWIN", "PageHinkley"]}
    explicit_drift_params = ParameterGrid(explicit_drift_param_grid)

    # Set up unlabeled drift detection params
    unlabeled_drift_param_grid = {   "allow_unlabeled_drift": [True],
                                    "unlabeled_drift_class": ["UnlabeledDriftDetector"],
                                    "explicit_drift_mode":["EnsembleDisagreement"]}
    unlabeled_drift_param = ParameterGrid(unlabeled_drift_param_grid)
    # Set up parameters:
    

    for param_set in unlabeled_drift_param:
        # This is an experiment
        
        # Load up configuration file
        mlepConfig = utils.load_json('./config/configuration/MLEPServer.json')

        # Update config file
        for _param in param_set:
            mlepConfig[_param] = param_set[_param]
    
    # We'll load thhe config file, make changes, and write a secondary file for experiments
    CONFIG_PATH = './config/configuration/ExperimentalConfig.json'
    with open(CONFIG_PATH, 'w') as write_:
        write_.write(json.dumps(mlepConfig))

    # Now we have the Experimental Coonfig we can use for running an experiment
    runExperiment(CONFIG_PATH)




def runExperiment(PATH_TO_CONFIG_FILE):

    # set up mlflow access

    # mlflow.set_tracking_uri -- not needed, defaults to mlruns

    # mlflow.create_experiment -- need experiment name. Should I programmatically create one? or go by timestamp

    mlflow.start_run()

    # log_param
    # Need to log current git id -- add this as a tag
    # Need to log entire configuration..

    internalTimer = 0
    streamData = StreamLocal.StreamLocal(data_source="data/data/2014_to_dec2018.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)

    augmentation = BatchedLocal.BatchedLocal(data_source='data/data/collectedIrrelevant.json', data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    augmentation.load_by_class()

    trainingData = BatchedLocal.BatchedLocal(data_source='data/data/initialTrainingData.json', data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    trainingData.load()

    # Now we have the data
    MLEPLearner = MLEPServer.MLEPLearningServer(PATH_TO_CONFIG_FILE)

    totalCounter = []
    mistakes = []
    while streamData.next():
        if internalTimer < streamData.getObject().getValue("timestamp"):
            internalTimer = streamData.getObject().getValue("timestamp")
            MLEPLearner.updateTime(internalTimer)

        classification = MLEPLearner.classify(streamData.getObject())


        totalCounter.append(1)
        if classification != streamData.getLabel():
            mistakes.append(1.0)
        else:
            mistakes.append(0.0)
        if len(totalCounter) % 1000 == 0 and len(totalCounter)>0:
            utils.std_flush("Completed", len(totalCounter), " samples, with running error (past 100) of", sum(mistakes[-100:])/sum(totalCounter[-100:]))
        if len(totalCounter) % 100 == 0 and len(totalCounter)>0:
            savePath.write(str(sum(mistakes[-100:])/sum(totalCounter[-100:]))+',')
    
    mlflow.end_run()    





if __name__ == "__main__":
    main()