import os, time, json, sys, pdb

import mlep.core.MLEPDriftAdaptor as MLEPDriftAdaptor
import mlep.data_model.BatchedLocal as BatchedLocal
import mlep.data_model.StreamLocal as StreamLocal
import mlep.data_set.PseudoJsonTweets as PseudoJsonTweets

import mlep.utils.io_utils as io_utils
import mlep.utils.time_utils as time_utils

import warnings
# warnings.filterwarnings(action="ignore", category=FutureWarning)

import traceback


def main():
    # set up the base config
    mlepConfig = io_utils.load_json("./MLEPServer.json")

    # update as per experiment requires
    mlepConfig["config"]["allow_model_confidence"] = True
    mlepConfig["config"]["allow_model_implicit_confidence"] = True

    mlepConfig["config"]["weight_method"] = "performance"
    mlepConfig["config"]["select_method"] = "recent"
    mlepConfig["config"]["filter_select"] = "nearest"

    # we are not updating internal timer...
    streamData = StreamLocal.StreamLocal(data_source="./data/realisticStreamComb_2013_feb19.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    

    augmentation = BatchedLocal.BatchedLocal(data_source="./data/collectedIrrelevant.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    augmentation.load_by_class()

    trainingData = BatchedLocal.BatchedLocal(data_source="./data/initialTrainingData.json", data_mode="single", data_set_class=PseudoJsonTweets.PseudoJsonTweets)
    trainingData.load()

    MLEPLearner = MLEPDriftAdaptor.MLEPDriftAdaptor(config_dict=mlepConfig, safe_mode=False)
    MLEPLearner.initialTrain(traindata=trainingData)
    io_utils.std_flush("Completed training at", time_utils.readable_time())
    MLEPLearner.addAugmentation(augmentation)
    io_utils.std_flush("Added augmentation at", time_utils.readable_time())

    totalCounter = 0
    implicit_mistakes = 0.0
    implicit_count = 0
    explicit_mistakes = 0.0
    explicit_count = 0

    while streamData.next():
        classification = MLEPLearner.classify(streamData.getObject(), classify_mode="implicit")
        if streamData.getLabel() is None:
            if classification != streamData.getObject().getValue("true_label"):
                implicit_mistakes += 1.0
            implicit_count += 1
        else:
            if classification != streamData.getLabel():
                explicit_mistakes += 1.0
            explicit_count += 1
        totalCounter += 1

        """
        if totalCounter % 100 == 0 and totalCounter>0.0:
            implicit_running_error = 2.00
            explicit_running_error = 2.00
            if implicit_count:
                implicit_running_error = implicit_mistakes/float(implicit_count)
            if explicit_count:
                explicit_running_error = explicit_mistakes/float(explicit_count)
            io_utils.std_flush("Fin: %6i samples\t\texplicit error: %2.4f\t\t implicit error: %2.4f"%(totalCounter, explicit_running_error, implicit_running_error))
            
            implicit_mistakes = 0.0
            implicit_count = 0
            explicit_mistakes = 0.0
            explicit_count = 0
        """
        

    
if __name__ == "__main__":
    main()