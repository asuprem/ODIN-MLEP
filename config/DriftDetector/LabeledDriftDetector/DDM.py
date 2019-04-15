from LabeledDriftDetector import LabeledDriftDetector


class DDM(LabeledDriftDetector):
    def __init__(self,min_instances=30, drift_level=3.0):

        from math import sqrt
        self.min_instances = min_instances
        self.drift_level = float(drift_level)
        self.i = None
        self.pi = None
        self.si = None
        self.pi_sd_min = None
        self.pi_min = None
        self.si_min = None
        self.sqrt=sqrt
        self.reset()


    
    def reset(self,):
        """ reset detector parameters, e.g. after drift has been detected """
        self.i = 0
        self.pi = 1.0
        self.si = 0.0
        #self.pi_sd_min = float("inf")
        self.pi_min = float("inf")
        self.si_min = float("inf")

    def detect(self,error):
        """

        error - 1 if classification was incorrect; 0 if classification was correct

        returns -- is there drift
        """
        #required for DDM

        self.i += 1
        self.pi = self.pi+ (error-self.pi)/float(self.i)
        self.si = self.sqrt(self.pi*  (1-self.pi)/self.i)

        if self.i < self.min_instances:
            return False

        if self.pi + self.si <= self.pi_min + self.si_min:
            self.pi_min = self.pi
            self.si_min = self.si
            #self.pi_sd_min = self.pi+self.si

        if self.pi + self.si > self.pi_min  + self.drift_level * self.si_min:
            return True
        else:
            return False

        
        

    