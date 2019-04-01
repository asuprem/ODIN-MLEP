
import os

class MLEPLearningServer():
    def __init__(self,):
        pass
        # This is the internal clock of the Server. Normally, this is time.time(). For this implementation, 
        # this is updated manually
        self.overallTimer = None

        # This is the clock for scheduled Filter Generation. During this scheduled generation, existing filters 
        # are also updated. Not yet sure how, but this is in progress
        self.scheduledFilterGenerateUpdateTimer = 0
        
        self.FIRST_TIMER = True

        self.SERVERTIME      = 'handshake/servertime'
        self.CREATETIME      = 'handshake/createtime'
        self.RECEIVEDTIME    = 'handshake/receivedtime'
        # For Drift based, models track their own 'drift'
        # ------------------------------------------------------------------------------------

    def updateOverallTimer(self):
        updateFlag = False
        while not updateFlag:
            #Check if server has received the time
            if not os.path.exists(self.CREATETIME):
                pass
            else:
                os.remove(self.CREATETIME)
                with open(self.SERVERTIME, 'r') as servertime:
                    self.overallTime = int(servertime.read().strip())
                open(self.RECEIVEDTIME, 'w').close()
                updateFlag = True
    
                if self.FIRST_TIMER:
                    # First time going through this. we will need to update all timers.
                    self.scheduledFilterGenerateUpdateTimer = self.overallTime
                    self.FIRST_TIMER = False
            

    def execute(self):

        while True:
            # Manually update the timer with info from application tester
            # We will do this by a handshaking protocol. So main application will create a SERVERTIME and write to it the server time. Then it will
            #       create the CREATETIME file
            # When server detects the CREATETIME file, it will read the SERVERTIME file and read the time. Then it will delete the CREATETIME file.
            #       Then it will create the RECEIVEDTIME file
            # application server detects RECEIVEDTIME file, updates SEVERTIME file, deletes RECEIVEDTIME file, and then creates CREATETIME file
            self.updateOverallTimer()

            #self.DoRestOfTheThings()
            if self.overallTime % 5000 == 0:
                print self.overallTime


'''
class MLEPPredictionServer():
    def __init__(self,):
        # Initialize Prediction Server

    def classify(self,data):
'''

if __name__ == "__main__":
    MLEP = MLEPLearningServer()
    MLEP.execute()