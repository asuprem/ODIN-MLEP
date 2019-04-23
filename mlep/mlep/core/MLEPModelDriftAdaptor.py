import os, time
import pdb
from mlep.utils import io_utils, sqlite_utils, time_utils
import sqlite3


class MLEPModelDriftAdaptor():
    def __init__(self, config_dict):
        """Initialize the learning server.

        config_dift -- [dict] JSON Configuration dictionary
        """
        io_utils.std_flush("Initializing MLEP_MODEL_DRIFT_ADAPTOR")

        import mlep.trackers.MetricsTracker as MetricsTracker
        import mlep.trackers.ModelDB as ModelDB
        import mlep.trackers.ModelTracker as ModelTracker

        self.setUpCoreVars()
        self.ModelDB = ModelDB.ModelDB()

        self.loadConfig(config_dict)
        self.initializeTimers()
        #self.setupDbConnection()
        #self.initializeDb()
        self.setUpEncoders()
        self.METRICS = MetricsTracker.MetricsTracker()
        self.setUpExplicitDriftTracker()
        self.setUpUnlabeledDriftTracker()
        self.setUpMemories()
        self.ModelTracker = ModelTracker.ModelTracker()

        io_utils.std_flush("Finished initializing MLEP...")

    def setUpCoreVars(self,):
        self.KNOWN_EXPLICIT_DRIFT_CLASSES = ["LabeledDriftDetector"]
        self.KNOWN_UNLABELED_DRIFT_CLASSES = ["UnlabeledDriftDetector"]

        # Setting of 'hosted' models + data cetroids
        self.MODELS = {}
        self.CENTROIDS={}

        # Augmenter
        self.AUGMENT = None

        # Statistics
        self.LAST_CLASSIFICATION = 0
        self.LAST_ENSEMBLE = []

        import sys
        self.HASHMAX = sys.maxsize




    def setUpUnlabeledDriftTracker(self,):
        if self.MLEPConfig["allow_unlabeled_drift"]:
            io_utils.std_flush("\tStarted setting up unlabeled drift tracker at", time_utils.readable_time())
            
            if self.MLEPConfig["unlabeled_drift_class"] not in self.KNOWN_UNLABELED_DRIFT_CLASSES:
                raise ValueError("Unlabeled drift class '%s' in configuration is not part of any known Unlabeled Drift Classes: %s"%(self.MLEPConfig["unlabeled_drift_class"], str(self.KNOWN_UNLABELED_DRIFT_CLASSES)))
            if self.MLEPConfig["unlabeled_drift_mode"] != "EnsembleDisagreement":
                raise NotImplementedError()
            driftTracker = self.MLEPConfig["unlabeled_drift_mode"]
            driftModule = self.MLEPConfig["unlabeled_drift_class"]
            driftArgs = self.MLEPConfig["drift_args"] if "drift_args" in self.MLEPConfig else {}
            driftModuleImport = __import__("mlep.drift_detector.%s.%s"%(driftModule, driftTracker), fromlist=[driftTracker])
            driftTrackerClass = getattr(driftModuleImport,driftTracker)
            self.UNLABELED_DRIFT_TRACKER = driftTrackerClass(**driftArgs)

            io_utils.std_flush("\tFinished setting up unlabeled drift tracker at", time_utils.readable_time())
        else:
            self.UNLABELED_DRIFT_TRACKER = None
            io_utils.std_flush("\tUnlabeled drift tracker not used in this run", time_utils.readable_time())

    def setUpExplicitDriftTracker(self,):
        if self.MLEPConfig["allow_explicit_drift"]:
            io_utils.std_flush("\tStarted setting up explicit drift tracker at", time_utils.readable_time())
            
            if self.MLEPConfig["explicit_drift_class"] not in self.KNOWN_EXPLICIT_DRIFT_CLASSES:
                raise ValueError("Explicit drift class '%s' in configuration is not part of any known Explicit Drift Classes: %s"%(self.MLEPConfig["explicit_drift_class"], str(self.KNOWN_EXPLICIT_DRIFT_CLASSES)))

            driftTracker = self.MLEPConfig["explicit_drift_mode"]
            driftModule = self.MLEPConfig["explicit_drift_class"]
            driftArgs = self.MLEPConfig["drift_args"] if "drift_args" in self.MLEPConfig else {}
            driftModuleImport = __import__("mlep.drift_detector.%s.%s"%(driftModule, driftTracker), fromlist=[driftTracker])
            driftTrackerClass = getattr(driftModuleImport,driftTracker)
            self.EXPLICIT_DRIFT_TRACKER = driftTrackerClass(**driftArgs)

            io_utils.std_flush("\tFinished setting up explicit drift tracker at", time_utils.readable_time())
        else:
            self.EXPLICIT_DRIFT_TRACKER = None
            io_utils.std_flush("\tExplicit drift tracker not used in this run", time_utils.readable_time())


    def loadConfig(self, config_dict):
        """Load JSON configuration file and initialize attributes.

        config_path -- [str] Path to the JSON configuration file.
        """
        io_utils.std_flush("\tStarted loading JSON configuration file at", time_utils.readable_time())

        self.config = config_dict
        if self.config["filter_select"] != "nearest":
            raise ValueError("MLEPModelDriftAdaptor requires nearest for filter_select")
        self.MLEPConfig = self.config["config"]
        self.MLEPModels = self.config["models"]
        self.MLEPPipelines = self.getValidPipelines()
        self.MLEPEncoders = self.getValidEncoders()

        io_utils.std_flush("\tFinished loading JSON configuration file at", time_utils.readable_time())

    def initializeTimers(self):
        """Initialize time attributes."""
        # Internal clock of the server.
        self.overallTimer = None

    def updateTime(self,timerVal):
        """ Manually updating time for experimental evaluation """
        self.overallTimer = timerVal



    def setUpEncoders(self):
        """Set up built-in encoders (Google News w2v)."""

        io_utils.std_flush("\tStarted setting up encoders at", time_utils.readable_time())

        
        self.ENCODERS = {}
        for _ , encoder_config in self.MLEPEncoders.items():
            io_utils.std_flush("\t\tSetting up encoder", encoder_config["name"], "at", time_utils.readable_time())
            encoderName = encoder_config["scriptName"]
            encoderModule = __import__("mlep.data_encoder.%s" % encoderName,
                    fromlist=[encoderName])
            encoderClass = getattr(encoderModule, encoderName)
            self.ENCODERS[encoder_config["name"]] = encoderClass()
            try:
                self.ENCODERS[encoder_config["name"]].setup(**encoder_config["args"])
            # Value Error is for joblib load -- need a message to convey as such
            except (IOError, ValueError) as e:
                io_utils.std_flush("Encoder load failed with error:", e, ". Attempting fix.")
                self.ENCODERS[encoder_config["name"]].failCondition(
                        **encoder_config["fail-args"])
                self.ENCODERS[encoder_config["name"]].setup(**encoder_config["args"])
                
        io_utils.std_flush("\tFinished setting up encoders at", time_utils.readable_time())

    def shutdown(self):
        # save models - because they are all heald in memory??
        # Access the save path
        # pick.dump models to that path
        pass
        self.ModelDB.close()
        

    def MLEPUpdate(self,memory_type="scheduled"):
        if self.MEMTRACK.memorySize(memory_name=memory_type) < self.MLEPConfig["min_train_size"]:
            io_utils.std_flush("Attemped update using", memory_type, "-memory with", self.MEMTRACK.memorySize(memory_name=memory_type),"data samples. Failed due to requirement of", self.MLEPConfig["min_train_size"], "samples." )    
            return
            # TODO update the learning model itself to reject update with too few? Or let user handle this issue?
        io_utils.std_flush("Update using", memory_type, "-memory at", time_utils.ms_to_readable(self.overallTimer), "with", self.MEMTRACK.memorySize(memory_name=memory_type),"data samples." )
        # Get the training data from Memory
        TrainingData = self.getTrainingData(memory_type=memory_type)
        self.MEMTRACK.clearMemory(memory_name=memory_type)
        
        # Generate
        self.train(TrainingData)
        io_utils.std_flush("Completed", memory_type, "-memory based Model generation at", time_utils.readable_time())

        # update
        self.update(TrainingData,models_to_update=self.MLEPConfig["models_to_update"])
        io_utils.std_flush("Completed", memory_type, "-memory based Model Update at", time_utils.readable_time())

        # Now we update model store.
        self.ModelTracker.updateModelStore(self.ModelDB)

    def getTrainingData(self, memory_type="scheduled"):
        """ Get the data in self.SCHEDULED_DATA_FILE """

        # need to load it as  BatchedModel...
        # (So, first scheduledDataFile needs to save stuff as BatchedModel...)

        # We load stuff from batched model
        # Then we check how many for each class
        # perform augmentation for the binary case
        import random
        scheduledTrainingData = None

        # TODO close the opened one before opening a read connection!!!!!
        scheduledTrainingData = self.MEMTRACK.transferMemory(memory_name = memory_type)
        scheduledTrainingData.load_by_class()

        if self.MEMTRACK.getClassifyMode() == "binary":
            negDataLength = scheduledTrainingData.class_size(0)
            posDataLength = scheduledTrainingData.class_size(1)
            if negDataLength < 0.8*posDataLength:
                io_utils.std_flush("Too few negative results. Adding more")
                if self.AUGMENT.class_size(0) < posDataLength:
                    # We'll need a random sampled for self.negatives BatchedLoad
                    scheduledTrainingData.augment_by_class(self.AUGMENT.getObjectsByClass(0), 0)
                else:
                    scheduledTrainingData.augment_by_class(random.sample(self.AUGMENT.getObjectsByClass(0), posDataLength-negDataLength), 0)
            elif negDataLength > 1.2 *posDataLength:
                # Too many negative data; we'll prune some
                io_utils.std_flush("Too many  negative samples. Pruning")
                scheduledTrainingData.prune_by_class(0,negDataLength-posDataLength)
                # TODO
            else:
                # Just right
                io_utils.std_flush("No augmentation necessary")
            # return combination of all classes
            return scheduledTrainingData
        else:
            raise NotImplementedError()


    def createPipeline(self,data, pipeline, source=None):
        """ Generate or Update a pipeline 
        
        If source is None, this is create. Else this is a generate.
        """

        # Data setup
        encoderName = pipeline["sequence"][0]

        X_train = self.ENCODERS[encoderName].batchEncode(data.getData())
        centroid = self.ENCODERS[encoderName].getCentroid(X_train)
        y_train = data.getLabels()

        # Model setup
        pipelineModel = pipeline["sequence"][1]

        pipelineModelName = self.MLEPModels[pipelineModel]["scriptName"]
        pipelineModelModule = __import__("mlep.learning_model.%s"%pipelineModelName, fromlist=[pipelineModelName])
        pipelineModelClass = getattr(pipelineModelModule,pipelineModelName)

        model = pipelineModelClass()

        precision, recall, score = None, None, None
        if source is None:
            # Generate
            pass
            precision, recall, score = model.fit_and_test(X_train, y_train)
        else:
            # Update
            model.clone(self.MODELS[source])
            precision, recall, score = model.update_and_test(X_train, y_train)

        return precision, recall, score, model, centroid

    

    def update(self, traindata, models_to_update='recent'):
        # for each model in self.MODELS
        # create a copy; rename details across everything
        # update copy
        # push details to DB
        prune_val = 5
        if self.MLEPConfig["update_prune"] == "C":
            # Keep constant to new
            prune_val = len(self.MLEPPipelines)
        else:
            raise NotImplementedError()
        
        temporaryModelStore = []
        modelSaveNames = [modelSaveName for modelSaveName in self.ModelTracker.get(models_to_update)]
        modelDetails = self.ModelDB.getModelDetails(modelSaveNames) # Gets fscore, pipelineName, modelSaveName
        pipelineNameDict = self.ModelDB.getDetails(modelDetails, 'pipelineName', 'dict')
        for modelSaveName in modelSaveNames:
            # copy model
            # set up new model
            
            # Check if model can be updated (some models cannot be updated)
            if not self.MODELS[modelSaveName].isUpdatable():
                continue

            currentPipeline = self.MLEPPipelines[pipelineNameDict[modelSaveName]]
            precision, recall, score, pipelineTrained, data_centroid = self.createPipeline(traindata, currentPipeline, modelSaveName)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"], score)
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath = ""
            testDataSavePath = ""
            
            # We temporarily load to dictionary for sorting later.
            dicta={}
            dicta["name"] = modelSavePath
            dicta["MODEL"] = pipelineTrained
            dicta["CENTROID"] = data_centroid
            dicta["modelid"] = modelIdentifier
            dicta["parentmodelid"] = str(modelSaveName)
            dicta["pipelineName"] = str(currentPipeline["name"])
            dicta["timestamp"] = timestamp
            dicta["data_centroid"] = data_centroid
            dicta["training_model"] = str(modelSavePath)
            dicta["training_data"] = str(trainDataSavePath)
            dicta["test_data"] = str(testDataSavePath)
            dicta["precision"] = precision
            dicta["recall"] = recall
            dicta["score"] = score
            dicta["_type"] = str(currentPipeline["type"])
            dicta["active"] = 1

            temporaryModelStore.append(dicta)

        if len(temporaryModelStore) > prune_val:
            io_utils.std_flush("Pruning models -- reducing from", str(len(temporaryModelStore)),"to",str(prune_val),"update models." )
            # keep the highest scoring update models
            temporaryModelStore = sorted(temporaryModelStore, key=lambda k:k["score"], reverse=True)
            temporaryModelStore = temporaryModelStore[:prune_val]

        for item in temporaryModelStore:
            # save the model (i.e. host it)
            item["MODEL"].trackDrift(self.MLEPConfig["allow_model_confidence"])
            self.MODELS[item["name"]] = item["MODEL"]
            # Because we are simplifying this implementation, we don't actually have pipeline families. Every pipelien is part of the w2v family
            # So we can actually just store data_centroids locally
            self.CENTROIDS[item["name"]] = item["data_centroid"]
            # Now we save deets.

            self.ModelDB.insertModelToDb(modelid=item["modelid"], parentmodelid=item["parentmodelid"], pipelineName=item["pipelineName"],
                                timestamp=item["timestamp"], data_centroid=item["data_centroid"], training_model=item["training_model"], 
                                training_data=item["training_data"], test_data=item["test_data"], precision=item["precision"], recall=item["recall"], score=item["score"],
                                _type=item["_type"], active=item["active"])



    # trainData is BatchedLocal
    def initialTrain(self,traindata,models= "all"):
        self.train(traindata)
        self.ModelTracker._set("train", self.ModelDB.getModelsSince())
        self.ModelTracker.updateModelStore(self.ModelDB)


    def train(self,traindata, models = 'all'):
        """ Function to train traindata """

        for pipeline in self.MLEPPipelines:
            # set up pipeline
            currentPipeline = self.MLEPPipelines[pipeline]
            precision, recall, score, pipelineTrained, data_centroid = self.createPipeline(traindata, currentPipeline)
            timestamp = time.time()
            modelIdentifier = self.createModelId(timestamp, currentPipeline["name"],score) 
            modelSavePath = "_".join([currentPipeline["name"], modelIdentifier])
            trainDataSavePath = ""
            testDataSavePath = ""

            # save the model (i.e. host it)
            pipelineTrained.trackDrift(self.MLEPConfig["allow_model_confidence"])
            self.MODELS[modelSavePath] = pipelineTrained
            # Because we are simplifying this implementation, we don't actually have pipeline families. Every pipelien is part of the w2v family
            # So we can actually just store data_centroids locally
            self.CENTROIDS[modelSavePath] = data_centroid
            del pipelineTrained
            # Now we save deets.
            # Some cleaning
            
            self.ModelDB.insertModelToDb(modelid=modelIdentifier, parentmodelid=None, pipelineName=str(currentPipeline["name"]),
                                timestamp=timestamp, data_centroid=data_centroid, training_model=str(modelSavePath), 
                                training_data=str(trainDataSavePath), test_data=str(testDataSavePath), precision=precision, recall=recall, score=score,
                                _type=str(currentPipeline["type"]), active=1)


    def createModelId(self, timestamp, pipelineName, fscore):
        strA = time_utils.time_to_id(timestamp)
        strB = time_utils.time_to_id(hash(pipelineName)%self.HASHMAX)
        strC = time_utils.time_to_id(fscore, 5)
        return "_".join([strA,strB,strC])
        
    
    def addAugmentation(self,augmentation):
        self.AUGMENT = augmentation

    
    def getValidPipelines(self,):
        """ get pipelines that are, well, valid """
        return {item:self.config["pipelines"][item] for item in self.config["pipelines"] if self.config["pipelines"][item]["valid"]}

    def getValidEncoders(self,):
        """ get valid encoders """
        # iterate through pipelines, get encoders that are valid, and return those from config->encoders
        return {item:self.config["encoders"][item] for item in {self.MLEPPipelines[_item]["encoder"]:1 for _item in self.MLEPPipelines}}

    def getValidModels(self,):
        """ get valid models """    
        ensembleModelNames = [item for item in self.ModelTracker.get(self.MLEPConfig["select_method"])]
        return ensembleModelNames


    def getTopKNearestModels(self,ensembleModelNames, data):
        # data is a DataSet object
        ensembleModelPerformance = None
        ensembleModelDistance = None
        # find top-k nearest centroids
        k_val = self.MLEPConfig["kval"]
        # don't need any fancy stuff if k-val is more than the number of models we have
        if k_val >= len(ensembleModelNames):
            pass
        else:

            # dictify for O(1) check
            ensembleModelNamesValid = {item:1 for item in ensembleModelNames}
            # 1. First, collect list of Encoders -- model mapping
            pipelineToModel = self.ModelDB.getPipelineToModel()
            
            # 2. Then create mapping of encoders -- model_save_path
            encoderToModel = {}
            for _pipeline in pipelineToModel:
                # Multiple pipelines can have the same encoder
                if self.MLEPPipelines[_pipeline]["sequence"][0] not in encoderToModel:
                    encoderToModel[self.MLEPPipelines[_pipeline]["sequence"][0]] = []
                # encoderToModel[PIPELINE_NAME] = [(MODEL_NAME, PERF),(MODEL_NAME, PERF)]
                encoderToModel[self.MLEPPipelines[_pipeline]["sequence"][0]] += pipelineToModel[_pipeline]
            
            # 3. Then for each encoder, find k-closest model_save_path
            kClosestPerEncoder = {}
            for _encoder in encoderToModel:
                kClosestPerEncoder[_encoder] = []
                _encodedData = self.ENCODERS[_encoder].encode(data.getData())
                # Find distance to all appropriate models
                # Then sort and take top-k
                # This can probably be optimized to not perform unneeded Distance calculations (if, e.g. two models have the same training dataset - something to consider)
                #   NOTE --> We need to make sure item[0] (modelName)
                #   NOTE --> item[1] : fscore
                #np.linalg.norm(_encodedData-self.CENTROIDS[item[0]])
                kClosestPerEncoder[_encoder]=[(self.ENCODERS[_encoder].getDistance(_encodedData, self.CENTROIDS[item[0]]), item[1], item[0]) for item in encoderToModel[_encoder] if item[0] in ensembleModelNamesValid]
                # Default sort on first param (norm); sort on distance - smallest to largest
                # tup[0] --> norm
                # tup[1] --> fscore
                # tup[2] --> modelName
                # TODO normalize distance to 0:1
                # Need to do this during centroid construction
                # for the training data, in addition to storing centroid, store furthest data point distance
                # Then during distance getting, we compare the distance to max_distance in getDistance() and return a 0-1 normalized. Anything outside max_distance is floored to 1.
                kClosestPerEncoder[_encoder].sort(key=lambda tup:tup[0])
                # Truncate to top-k
                kClosestPerEncoder[_encoder] = kClosestPerEncoder[_encoder][:k_val]

            # 4. Put them all together and sort on performance
            # distance weighted performance
            kClosest = []
            for _encoder in kClosestPerEncoder:
                kClosest+=kClosestPerEncoder[_encoder]
            # Sorting by tup[1] --> fscore
            kClosest.sort(key=lambda tup:tup[1], reverse=True)

            # 5. Return top-k (so two levels of k, finally returning k models)
            # item[0] --> norm
            # item[1] --> fscore
            # item[2] --> modelName
            kClosest = kClosest[:k_val]
            ensembleModelNames = [None]*k_val
            ensembleModelDistance = [None]*k_val
            ensembleModelPerformance = [None]*k_val
            for idx,_item in enumerate(kClosest):
                ensembleModelNames[idx] = _item[2]
                ensembleModelPerformance[idx] = _item[1]
                ensembleModelDistance[idx] = _item[0]
        return ensembleModelNames, ensembleModelPerformance, ensembleModelDistance

    def classify(self, data, classify_mode="explicit"):          
        """
        MLEPModelDriftAdaptor's classifier. 
        
        The MLEPModelDriftAdaptor's classifier performs model retrieval, distance calculation, and drift updates.

        Args:
            data -- a single data sample for classification
            classify_mode -- "explicit" if data is supposed to have a label. "implicit" if data is unlabeled. Since this is an experimentation framework, data is technically supposed to have a label. The distinction is on whether the label is something MLEPModelDriftAdaptor would have access to during live operation. "explicit" refers to this case, while "implicit" refers to the verso.

        Returns:
            classification -- INT -- currently only binary classification is supported
            
        """
        
        
        # First set up list of correct models
        ensembleModelNames = self.getValidModels()
        # Now that we have collection of candidaate models, we use filter_select to decide how to choose the right model
        # self.MLEPConfig["filter_select"] == "nearest":
        ensembleModelNames, ensembleModelPerformance, ensembleModelDistance = self.getTopKNearestModels(ensembleModelNames, data)

        # Given ensembleModelNames, use all of them as part of ensemble
        # Run the sqlite query to get model details
        modelDetails = self.ModelDB.getModelDetails(ensembleModelNames)
        if self.MLEPConfig["weight_method"] == "performance":
            if ensembleModelPerformance is not None:
                weights = ensembleModelPerformance
            else:
                # request DB for performance (f-score)
                weights = self.ModelDB.getDetails(modelDetails, 'fscore', 'list', order=ensembleModelNames)
            sumWeights = sum(weights)
            weights = [item/sumWeights for item in weights]
        elif self.MLEPConfig["weight_method"] == "unweighted":
            weights = len(ensembleModelNames)*[1.0/len(ensembleModelNames)]
        else:
            weights = len(ensembleModelNames)*[1.0/len(ensembleModelNames)]
        

        # Get encoder types in ensembleModelNames                       
        # build local dictionary of data --> encodedVersion             
        pipelineNameDict = self.ModelDB.getDetails(modelDetails, 'pipelineName', 'dict')
        localEncoder = {}
        for modelName in pipelineNameDict:
            pipelineName = pipelineNameDict[modelName]
            localEncoder[self.MLEPPipelines[pipelineName]["sequence"][0]] = 0
        
        for encoder in localEncoder:
            localEncoder[encoder] = self.ENCODERS[encoder].encode(data.getData())


        #---------------------------------------------------------------------
        # Time to classify
        
        classification = 0
        ensembleWeighted = [0]*len(ensembleModelNames)
        ensembleRaw = [0]*len(ensembleModelNames)
        for idx,_name in enumerate(ensembleModelNames):
            # use the prescribed enc; ensembleModelNames are the modelSaveFile
            # We need the pipeline each is associated with (so that we can extract front-loaded encoder)
            locally_encoded_data = localEncoder[self.MLEPPipelines[pipelineNameDict[_name]]["sequence"][0]]
            # So we get the model name, access the pipeline name from pipelineNameDict
            # Then get the encodername from sequence[0]
            # Then get the locally encoded thingamajig of the data
            # And pass it into predict()
            cls_ = None
            # for regular mode
            if not self.MLEPConfig["allow_model_confidence"]:
                cls_=self.MODELS[_name].predict(locally_encoded_data)
            
            else:
                # TODO for model-drift mode, we first check if the data point is ground-truth stream or a prediction stream
                #  -- if getLabel() on data returns None, it is a prediction stream
                # -- if getLabel() on data returns a value, it is a ground-truth stream
                
                # TODO getLabel() returns value
                # if explicit_drift_mode is allowed
                # cls = self.MODELS[_name].predict("explicit", y_label = label) -- > this will perform drift detection internally
                if classify_mode == "explicit":
                    # TODO TODO TODO ALERT ALERT ALERT NOT FUNCTIONAL
                    cls_=self.MODELS[_name].predict(locally_encoded_data, mode="explicit", y_label = data.getLabel())



                # TODO getLabel() returns None
                # if implicit drift mode is allowed
                # cls = self.MODELS[_name].predict("implicit") --> also perform drift detection internally
                if classify_mode == "implicit":
                    # TODO TODO TODO ALERT ALERT ALERT NOT FUNCTIONAL
                    cls_=self.MODELS[_name].predict(locally_encoded_data, mode="implicit")

            ensembleWeighted[idx] = float(weights[idx]*cls_)
            ensembleRaw[idx] = float(cls_)

        # Assume binary. We'll deal with others later
        classification = sum(ensembleWeighted)
        classification =  0 if classification < 0.5 else 1
        

        error = 1 if classification != data.getLabel() else 0
        ensembleError = [(1 if ensembleRawScore != data.getLabel() else 0) for ensembleRawScore in ensembleRaw]

        self.METRICS.updateMetrics(classification, error, ensembleError, ensembleRaw, ensembleWeighted)

        # add to scheduled memory if this is explicit data
        
        if self.MLEPConfig["allow_update_schedule"]:
            if classify_mode == "explicit":
                self.MEMTRACK.addToMemory(memory_name="scheduled", data=data)
                if error:
                    self.MEMTRACK.addToMemory(memory_name="scheduled_errors", data=data)
                # No drift detection necessary
                # No MLEPUpdate necessary

        # perform explicit drift detection and update (if classift mode is explicit)
        if self.MLEPConfig["allow_explicit_drift"]:
            # send the input appropriate for the drift mode
            # shuld be updated to be more readble; update so that users can define their own drift tracking method
            if classify_mode == "explicit":
                # add error -- 
                self.MEMTRACK.addToMemory(memory_name="explicit_drift", data=data)
                if error:
                    self.MEMTRACK.addToMemory(memory_name="explicit_errors", data=data)

                driftDetected = self.EXPLICIT_DRIFT_TRACKER.detect(self.METRICS.get(self.MLEPConfig["drift_metrics"][self.MLEPConfig["explicit_drift_mode"]]))
                if driftDetected:
                    io_utils.std_flush(self.MLEPConfig["explicit_drift_mode"], "has detected drift at", len(self.METRICS.get("all_errors")), "samples. Resetting")
                    self.EXPLICIT_DRIFT_TRACKER.reset()
                    
                    # perform drift update (big whoo)
                    # perform update with the correct memory type
                    if self.MLEPConfig["explicit_update_mode"] == "all":
                        self.MLEPUpdate(memory_type="explicit_drift")
                    elif self.MLEPConfig["explicit_update_mode"] == "errors":
                        self.MLEPUpdate(memory_type="explicit_errors")
                    else:
                        raise NotImplementedError()
                

        # perform implicit/unlabeled drift detection and update. This is performed :
        if self.MLEPConfig["allow_unlabeled_drift"]:
            # send the input appropriate for the drift mode
            # shuld be updated to be more readble; update so that users can define their own drift tracking method
            # Add to memory only if explicit, else perform drift detection
            if classify_mode == "explicit":
                self.MEMTRACK.addToMemory(memory_name="unlabeled_drift", data=data)
                if error:
                    self.MEMTRACK.addToMemory(memory_name="unlabeled_errors", data=data)
            if classify_mode == "implicit":
                if self.MLEPConfig["unlabeled_drift_mode"] == "EnsembleDisagreement":

                    driftDetected = self.UNLABELED_DRIFT_TRACKER.detect(self.METRICS.get(self.MLEPConfig["drift_metrics"][self.MLEPConfig["unlabeled_drift_mode"]]))
                    if driftDetected:
                        io_utils.std_flush(self.MLEPConfig["unlabeled_drift_mode"], "has detected drift at", len(self.METRICS.get("all_errors")), "samples. Resetting")
                        self.UNLABELED_DRIFT_TRACKER.reset()
                        
                        # perform drift update (big whoo)
                        # perform update with the correct memory type
                        # TODO uncomment this -- for now we are just checking if drift is being detected....
                        """
                        if self.MLEPConfig["unlabeled_update_mode"] == "all":
                            self.MLEPUpdate(memory_type="unlabeled_drift")
                        elif self.MLEPConfig["unlabeled_update_mode"] == "errors":
                            self.MLEPUpdate(memory_type="unlabeled_errors")
                        else:
                            raise NotImplementedError()
                        """
                else:
                    raise NotImplementedError()

        if self.MLEPConfig["allow_model_confidence"]:
            # TODO
            for idx,_name in enumerate(ensembleModelNames):
                # add data to proper memory (core-mem, gen-mem)
                # add data, if it doesn't fit either, to data-mem (from MEMORY_TRACK)
                pass

                # check if model is drifting
                # if so use core-mem and gen-mem to update the model.
                pass
                modelDrifting = self.MODELS[_name].isDrifting()
                if modelDrifting:
                    io_utils.std_flush(_name, "has detected drift at", len(self.METRICS.get("all_errors")), "samples. Resetting")
                    
            # TODO for this, for now, just output size of data-mem. See if this changes significantly, and use heuristics ?????
            # TODO Check if there is -- explicit drift -- OR -- enough data in data-mem to update
            # If so, generate new models on data-mem and add them to the pile
            # TODO TODO TODO Better way to check --> if more and more unlabeled samples are close to data-mem, then strengthen data-mem. 
        
        self.saveClassification(classification)
        self.saveEnsemble(ensembleModelNames)

        return classification


    def saveClassification(self, classification):
        self.LAST_CLASSIFICATION = classification
    def saveEnsemble(self,ensembleModelNames):
        self.LAST_ENSEMBLE = [item for item in ensembleModelNames]


    def setUpMemories(self,):

        io_utils.std_flush("\tStarted setting up memories at", time_utils.readable_time())
        import mlep.trackers.MemoryTracker as MemoryTracker
        self.MEMTRACK = MemoryTracker.MemoryTracker()

        if self.MLEPConfig["allow_update_schedule"]:
            self.MEMTRACK.addNewMemory(memory_name="scheduled",memory_store="memory")
            self.MEMTRACK.addNewMemory(memory_name="scheduled_errors",memory_store="memory")
            io_utils.std_flush("\t\tAdded scheduled memory")

        if self.MLEPConfig["allow_explicit_drift"]:
            self.MEMTRACK.addNewMemory(memory_name="explicit_drift",memory_store="memory")
            self.MEMTRACK.addNewMemory(memory_name="explicit_errors",memory_store="memory")
            io_utils.std_flush("\t\tAdded explicit drift memory")

        if self.MLEPConfig["allow_unlabeled_drift"]:
            self.MEMTRACK.addNewMemory(memory_name="unlabeled_drift",memory_store="memory")
            self.MEMTRACK.addNewMemory(memory_name="unlabeled_errors",memory_store="memory")
            io_utils.std_flush("\t\tAdded unlabeled drift memory")
        
        io_utils.std_flush("\tFinished setting up memories at", time_utils.readable_time())


"""
{
    "name": "Python: SimpleExperiment",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/mlep/experiments/single_experiment.py",
    "console": "integratedTerminal",
    "args": [
        "test",
        "--allow_explicit_drift", "True",
        "--allow_update_schedule", "True"
    ],
    "cwd":"${workspaceFolder}/mlep/experiments/"
},
"""