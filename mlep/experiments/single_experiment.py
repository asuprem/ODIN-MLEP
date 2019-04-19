import os, time, json, sys, pdb, click

import mlflow

import mlep.core.MLEPServer as MLEPServer

import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets
import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

"""
Arguments

python application.py experimentName [updateSchedule] [weightMethod] [selectMethod] [filterMethod] [kVal]

"""
LOG_FILE = "./logfiles/application.log"

@click.command()
@click.argument('experimentname')
@click.option('--update', type=int)
@click.option('--weights', type=click.Choice(["unweighted", "performance"]))
@click.option('--select', type=click.Choice(["train", "historical", "historical-new", "historical-updates","recent","recent-new","recent-updates"]))
@click.option('--filter', type=click.Choice(["no-filter", "top-k", "nearest"]))
@click.option('--kval', type=int)
def main(experimentname, update, weights, select, filter, kval):
    # set up logging
    if not os.path.exists('./logfiles'):
        os.makedirs('logfiles')

    sys.stdout = open(LOG_FILE, "w")

    # Tracking URI -- yeah it's not very secure, but w/e
    mlflow.set_tracking_uri("mysql://mlflow:mlflow@127.0.0.1:3306/mlflow_runs")
    # Where to save data:
    mlflow.start_run(run_name=experimentname)


    # We'll load thhe config file, make changes, and write a secondary file for experiments
    mlepConfig = io_utils.load_json('./MLEPServer.json')

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
    
    PATH_TO_CONFIG_FILE = './ExperimentalConfig.json'
    with open(PATH_TO_CONFIG_FILE, 'w') as write_:
        write_.write(json.dumps(mlepConfig))

    
    # Log relevant details
    for _key in mlepConfig["config"]:
        # possible error
        if _key != "drift_metrics":
            mlflow.log_param(_key, mlepConfig["config"][_key])
    

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
    
    mlflow.log_param("run_complete", True)
    mlflow.log_param("total_samples", totalCounter)  
    mlflow.log_artifact(LOG_FILE)
    mlflow.end_run()

if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter