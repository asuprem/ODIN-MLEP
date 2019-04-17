import UnlabeledDriftDetector


class EnsembleDisagreement(UnlabeledDriftDetector.UnlabeledDriftDetector):
    def __init__(self,threshold=0.8):

        from itertools import combinations
        self.threshold = threshold
        self.reset()
        self.combinations = combinations


    
    def reset(self,):
        """ reset detector parameters, e.g. after drift has been detected """

        pass

    def detect(self,ensembleClassification):
        """

        nsembleClassification - list of classifications

        returns -- is there drift
        """
        #required for DDM
        import pdb
        pdb.set_trace()   
        comp = [item for item in self.combinations(ensembleClassification, 2)] 
        delta = [0]*len(comp)
        delta_sum = 0.0
        idx_
        for idx,scoreTuple in enumerate(comp):
            delta[idx] = float(scoreTuple[0] == scoreTuple[1])
            delta_sum+=delta[idx]
            idx_=idx

        delta_avg = delta_sum/float(idx_)

        if delta_avg > self.threshold:
            return True
        else:
            return False

            
        

    