
class OnlineSimilarityDistribution:
    def __init__(self, nBins):
        #self.dist = {item*(1./nBins):0 for item in range(1,nBins+1)}
        self.dist = [0 for item in range(1,nBins+1)]
        self.dist_keys = [item*(1./nBins) for item in range(1,nBins+1)]
        import bisect
        self.bisect = bisect.bisect
        self.max_len = 0.0


    def _findIndex(self,data):
        i = self.bisect(self.dist_keys, data)
        return i

    def get(self,data):
        self.max_len+=1.0
        if data >= 1.0:
            return self.dist[1.0]/self.max_len
        else:
            return self.dist[self._findIndex(data)]/self.max_len

    def update(self,data):
        self.max_len+=1.0
        if data >= 1.0:
            self.dist[-1] += 1
        else:
            self.dist[self._findIndex(data)] += 1
        