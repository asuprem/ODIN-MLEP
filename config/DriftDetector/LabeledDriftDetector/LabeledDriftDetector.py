__metaclass__ = type

class LabeledDriftDetector():
    """ This is a labeled drift detector. It takes as input mistakes, and detects if drift has occured """


    def __init__(self,):
        raise NotImplementedError()

    def reset(self,):
        """ reset detector parameters, e.g. after drift has been detected """
        raise NotImplementedError()

    def detect(self,classification):
        """

        classificaation - 1 if classification was correct; 0 if classification was wrong

        returns -- is there drift
        """

        raise NotImplementedError()

    