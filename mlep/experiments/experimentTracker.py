import os, time, json, sys, pdb, click

import mlep.core.MLEPServer as MLEPServer

import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets
import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils


import mlflow

from sklearn.model_selection import ParameterGrid

import traceback

LOG_FILE = "./logfiles/application.log"
EXP_STATUS = "./logfiles/experiments.log"
def main():

    exp_status_write = open(EXP_STATUS, "a")
    exp_status_write.write("\n\n\n\n")
    exp_status_write.write("--------------------------------------")
    exp_status_write.write("  BEGINNING NEW EXECUTION AT " + str(time_utils.readable_time("%Y-%m-%d %H:%M:%S"))) 
    exp_status_write.write("  --------------------------------------"+ "\n\n")
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
    scheduled_param = ParameterGrid(scheduled_param_grid)

    # Set up explicit drift detection params
    explicit_drift_param_grid = {   "allow_explicit_drift": [(True,"ExpDr")],
                                    "explicit_drift_class": [("LabeledDriftDetector","LDD")],
                                    "explicit_drift_mode":[("DDM","DDM"), ("EDDM","EDDM"), ("ADWIN","ADWIN"), ("PageHinkley", "PageHinkley")], 

                                    "allow_unlabeled_drift": [(False,"")],
                                    "allow_update_schedule": [(False,"")],

                                    "weights":[( "unweighted","U"),( "performance","P" )],
                                    "select":[(  "recent","RR" ) , ( "recent-new","RN" ) , ( "recent-updates","RU" ) , ( "historical-new","HN" ) , ( "historical-updates","HU" ) , ( "historical","HH" )],
                                    "filter":[("no-filter","F") , ("top-k","T"),("nearest","N")],
                                    "kval":[("5","5"), ("10","10")]}
    explicit_drift_params = ParameterGrid(explicit_drift_param_grid)

    # Set up unlabeled drift detection params
    unlabeled_drift_param_grid = {   "allow_unlabeled_drift": [True],
                                    "unlabeled_drift_class": ["UnlabeledDriftDetector"],
                                    "explicit_drift_mode":["EnsembleDisagreement"]}
    unlabeled_drift_param = ParameterGrid(unlabeled_drift_param_grid)
    # Set up parameters:
    

    for param_set in explicit_drift_params:
        # This is an experiment
        
        # Load up configuration file
        mlepConfig = io_utils.load_json('./MLEPServer.json')

        # Update config file and generate an experiment name
        experiment_name=''
        for _param in param_set:
            if param_set[_param][1] != "":
                experiment_name+=param_set[_param][1] + '-'
            mlepConfig["config"][_param] = param_set[_param][0]
        experiment_name = experiment_name[:-1]
    
        # We'll load thhe config file, make changes, and write a secondary file for experiments
        CONFIG_PATH = './ExperimentalConfig.json'
        with open(CONFIG_PATH, 'w') as write_:
            write_.write(json.dumps(mlepConfig))

        # Now we have the Experimental Coonfig we can use for running an experiment
        # generate an experiment name
        exp_status_write.write("--STATUS-- " + experiment_name + "   ")
        try:
            runExperiment(CONFIG_PATH, mlepConfig, experiment_name)
            exp_status_write.write("SUCCESS\n")
        except Exception as e:
            exp_status_write.write("FAILED\n")
            exp_status_write.write(traceback.format_exc())
            exp_status_write.write(str(e))
            exp_status_write.write("\n")

    exp_status_write.close()




def runExperiment(PATH_TO_CONFIG_FILE, mlepConfig, experiment_name):

    # set up mlflow access
    # mlflow.set_tracking_uri -- not needed, defaults to mlruns
    # mlflow.create_experiment -- need experiment name. Should I programmatically create one? or go by timestamp
    sys.stdout = open(LOG_FILE, "w")

    mlflow.set_tracking_uri("mysql://mlflow:mlflow@127.0.0.1:3306/mlflow_runs")
    mlflow.start_run(run_name="explicit_drift_analysis")

    # Log relevant details
    for _key in mlepConfig["config"]:
        # possible error
        if _key != "drift_metrics":
            mlflow.log_param(_key, mlepConfig["config"][_key])
    mlflow.log_param("experiment_name", experiment_name)


    internalTimer = 0
    streamData = StreamLocal.StreamLocal(data_source="data/2014_to_dec2018.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)

    augmentation = BatchedLocal.BatchedLocal(data_source='data/collectedIrrelevant.json', data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    augmentation.load_by_class()

    trainingData = BatchedLocal.BatchedLocal(data_source='data/initialTrainingData.json', data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    trainingData.load()

    # Now we have the data
    MLEPLearner = MLEPServer.MLEPLearningServer(PATH_TO_CONFIG_FILE)

    # Perform initial traininig
    MLEPLearner.initialTrain(traindata=trainingData)
    io_utils.std_flush("Completed training at", time_utils.readable_time())
    MLEPLearner.addAugmentation(augmentation)
    io_utils.std_flush("Added augmentation at", time_utils.readable_time())

    totalCounter = 0.0
    mistakes = []
    while streamData.next():
        if internalTimer < streamData.getObject().getValue("timestamp"):
            internalTimer = streamData.getObject().getValue("timestamp")
            MLEPLearner.updateTime(internalTimer)

        classification = MLEPLearner.classify(streamData.getObject())


        totalCounter += 1.0
        if classification != streamData.getLabel():
            mistakes.append(1.0)
        else:
            mistakes.append(0.0)
        if totalCounter % 1000 == 0 and totalCounter>0.0:
            io_utils.std_flush("Completed", int(totalCounter), " samples, with running error (past 100) of", sum(mistakes[-100:])/100.0)
        if totalCounter % 100 == 0 and totalCounter>0.0:
            running_error = sum(mistakes[-100:])/100.0
            mlflow.log_metric("running_err"+str(int(totalCounter/100)), running_error)
    

    MLEPLearner.shutdown()

    io_utils.std_flush("\n-----------------------------\nCOMPLETED\n-----------------------------\n")
    
    sys.stdout.close()
    
    mlflow.log_param("total_samples", totalCounter)  
    mlflow.log_artifact(LOG_FILE)
    mlflow.log_param("run_complete", True)
    mlflow.end_run()




if __name__ == "__main__":
    main()