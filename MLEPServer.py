

class MLEPLearningServer():
    def __init__(self,):
        pass
        # This is the internal clock of the Server. Normally, this is time.time(). For this implementation, 
        # this is updated manually
        self.overallTimer

        # This is the clock for scheduled Filter Generation. During this scheduled generation, existing filters 
        # are also updated. Not yet sure how, but this is in progress
        self.scheduledFilterGenerateUpdateTimer

        # For Drift based, models track their own 'drift'
        # ------------------------------------------------------------------------------------

        
        

    def execute(self):

        while True:
            # Manually update the timer with info from application tester
            # We will do this by a handshaking protocol. So main application will create a SERVERTIME and write to it the server time. Then it will
            #       create the CREATETIME file
            # When server detects the CREATETIME file, it will read the SERVERTIME file and read the time. Then it will delete the CREATETIME file.
            #       Then it will create the RECEIVEDTIME file
            # application server detects RECEIVEDTIME file, updates SEVERTIME file, deletes RECEIVEDTIME file, and then creates CREATETIME file
            self.updateOverallTimer()

            self.DoRestOfTheThings()



class MLEPPredictionServer():
    def __init__(self,):
        # Initialize Prediction Server

    def classify(self,data):