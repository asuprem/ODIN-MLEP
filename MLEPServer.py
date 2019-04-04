
import os, json, pdb, codecs
import sqlite3
from sqlite3 import Error
import shutil
from utils import *

class MLEPLearningServer():
    def __init__(self,):
        
        std_flush("Initializing")


        """ This is the internal clock of the Server. Normally, this is time.time(). For this implementation, this is updated manually """
        self.overallTimer = None

        """ This is the clock for scheduled Filter Generation. During this scheduled generation, existing filters are also updated. Not yet sure how, but this is in progress """
        self.scheduledFilterGenerateUpdateTimer = 0
        self.DAY_IN_MS = 86400000
        self.scheduledSchedule =  self.DAY_IN_MS * 30

        std_flush("Finished up timers at ", readable_time())
        # For Drift based, models track their own 'drift'
        # ------------------------------------------------------------------------------------


        """ Set up storage directories """
        self.SOURCE_DIR = './.MLEPServer'
        self.setups = ['models', 'data', 'modelSerials', 'db']
        self.DB_FILE = './.MLEPServer/db/MLEP.db'
        self.SCHEDULED_DATA_FILE = './.MLEPServer/data/scheduledFile.json'
        std_flush("Finished up path variables at ", readable_time())
        
        """ create scheduled file """
        open(self.SCHEDULED_DATA_FILE, 'w').close()
        std_flush("Created data file at", readable_time())

        try:
            shutil.rmtree(self.SOURCE_DIR)
        except:
            pass
        os.makedirs(self.SOURCE_DIR)
        for directory in self.setups:
            os.makedirs(os.path.join(self.SOURCE_DIR, directory))

        std_flush("Finished setting up directory structure at", readable_time())

        """ create Database Connections and perform initial setup """
        self.DB_CONN = self.createDBConnection()
        self.initializeDB()
        std_flush("Initialized DB at ", readable_time())

        # This would normally be a set of hosted encoders. For local implementation, we have the encoders as a dict of encoder objects (TODO)
        std_flush("Setting up built-in encoders", readable_time())
        self.ENCODERS = {}
        self.MODELS = {}
        self.setUpEncoders()


    def setUpEncoders(self):
        """ This sets up built-in encoders. For now, this is all there is. Specifically, we only have pretrained Google News w2v """
        configFile = self.load_json('config/MLEPServer.json')
        # Load Encoder configurations
        for encoders in configFile["encoders"]:
            # For each encoder, load it first
            currentEncoder = configFile["encoders"][encoders]

            std_flush("Setting up", currentEncoder["name"], "at", readable_time())
            
            encoderName = currentEncoder["scriptName"]
            encoderModule = __import__("config.DataEncoder.%s"%encoderName, fromlist=[encoderName])
            encoderClass = getattr(encoderModule,encoderName)

            # Set up encoder(s)
            self.ENCODERS[currentEncoder["name"]] = encoderClass()

        
    def createDBConnection(self,):
        """ create a database connection to a SQLite database """
        try:
            conn = sqlite3.connect(self.DB_FILE)
            #print(sqlite3.version)
        except Error as e:
            print(e)
        return conn


    def closeDBConnection(self,):
        try:
            self.DB_CONN.close()
        except:
            pass
    
    def initializeDB(self):
        """ Initialize tables in database """
        # Initialize Model table
        self.DB_CONN.execute("""CREATE TABLE IF NOT EXISTS Models
                                (   modelid         integer primary key autoincrement,
                                    name            text, 
                                    timestamp       real,
                                    data_centroid   text,
                                    data            text,
                                    trainingModel   text,
                                    trainingData    text,
                                    testData        text,
                                    precision       real,
                                    recall          real,
                                    fscore          real,
                                    type            text )""")
    
    def updateTime(self,timerVal):
        """ Manually updating time for experimental evaluation """

        self.overallTimer = timerVal
        
        # Check scheduled time difference if there need to be updates
        if abs(self.overallTimer - self.scheduledFilterGenerateUpdateTimer) > self.scheduledSchedule:
            if not os.path.exists(self.SCHEDULED_DATA_FILE):
                # Something is the issue
                std_flush("No data for update")
                self.scheduledFilterGenerateUpdateTimer = self.overallTimer
            else:    
                # perform scheduled update
                
                # show lines in file
                num_lines = sum(1 for line in open(self.SCHEDULED_DATA_FILE))

                std_flush("Scheduled update at", ms_to_readable(self.overallTimer), "with", num_lines,"data samples." )
                self.scheduledFilterGenerateUpdateTimer = self.overallTimer
                
                # delete file
                '''
                try:
                    os.remove(self.SCHEDULED_DATA_FILE)
                except:
                    pass
                '''
                open(self.SCHEDULED_DATA_FILE, 'w').close()


    def generatePipeline(self,data, pipeline):
        """ Generate a model using provided pipeline """
        
        # Simplified pipeline. First entry is Encoder; Second entry is the actual Model
        configFile = self.load_json('config/MLEPServer.json')

        encoderName = pipeline["sequence"][0]
        pipelineModel = pipeline["sequence"][1]

        # Perform lookup
        pipelineModelName = configFile["models"][pipelineModel]["scriptName"]
        pipelineModelModule = __import__("config.LearningModel.%s"%pipelineModelName, fromlist=[pipelineModelName])
        pipelineModelClass = getattr(pipelineModelModule,pipelineModelName)

        model = pipelineModelClass()
        tst2=self.ENCODERS['w2v-main'].batchEncode([item['text'] for item in data])
        tst3 = [item['label'] for item in data]
        model.fit(X=tst2,y=tst3)

        pdb.set_trace()
        pass
        return precision, recall, score, model

    

    def train(self,traindata, models = 'all'):
        # for each modelType in modelTypes
        #   for each encodingType (just 1)
        #       Create sklearn model using default details
        #       then train sklearn model using encoded data
        #       precision, recall, score, model = self.generate(encoder, traindata, model)
        #       push details to ModelDB
        #       save model to file using ID as filename.model -- serialized sklearn model
        
        

        # First load the Model configurations - identify what models exist
        configFile = self.load_json('config/MLEPServer.json')
        
        for pipeline in configFile["pipelines"]:
            
            
            # We make the simplified assumption that all encoders are the same (pretrained w2v). 
            # So we don't have to handle pipeline families at this point for the distance function (if implemented)
            # Also, since our models are small-ish, we can make do by hosting models in memory
            # Production implementation (and going forward), models would be hosted as an API endpoint until "retirement"

            #std_flush("Setting up", currentEncoder["name"], "at", readable_time())
            
            # set up pipeline
            currentPipeline = configFile["pipelines"][pipeline]
            precision, recall, score, pipelineTrained = self.generatePipeline(traindata, currentPipeline)



        

        pass


    def load_json(self,json_):
        return json.load(codecs.open(json_, encoding='utf-8'))

    '''
    def addNegatives(self,negatives):
    '''






class MLEPPredictionServer():
    def __init__(self,):
        # Initialize Prediction Server
        # Set up storage directories
        self.SOURCE_DIR = './.MLEPServer'
        self.setups = ['models', 'data', 'modelSerials', 'db']
        
        self.SCHEDULED_DATA_FILE = './.MLEPServer/data/scheduledFile.json'
        self.CLASSIFY_MODE = 'knn'
        self.SCHEDULED_DATA_FILE_OPERATOR = open(self.SCHEDULED_DATA_FILE, 'a')

    def setMode(self,mode):
        if mode == 'knn' or mode == 'recent':
            self.CLASSIFY_MODE = mode
        else:
            self.CLASSIFY_MODE = 'recent'

    def classify(self,data):
        # sve data item to scheduledDataFile
        try:
            self.SCHEDULED_DATA_FILE_OPERATOR.write(json.dumps(data)+'\n')
        except:
            self.SCHEDULED_DATA_FILE_OPERATOR = open(self.SCHEDULED_DATA_FILE, 'a')
            self.SCHEDULED_DATA_FILE_OPERATOR.write(json.dumps(data)+'\n')

        if self.CLASSIFY_MODE == 'recent':
            # get more recent created models by timestamp
            pass


        




'''
if __name__ == "__main__":
    MLEP = MLEPLearningServer()
    MLEP.execute()
'''