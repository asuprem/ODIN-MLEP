import numpy as np

class L2NormDataCharacteristics:
    def __init__(self, nBins=40, alpha = 0.6):
        self.nBins = nBins
        self.distribution = []
        self.alpha = alpha
        self.attrs={}
        self.attrs["centroid"] = None
        self.attrs["delta_low"] = None
        self.attrs["delta_high"] = None
        
        self.dist_keys = [item*(1./self.nBins) for item in range(self.nBins+1)]
        self.dist = None

    def buildDistribution(self,centroid,data):
        """ Build the data distribution given a data set and its centroid.

        Uses cosine_similary metric to build the characteristics 'map' of a text-based dataset

        """

        self.attrs["centroid"] = centroid
        self.distribution = [0]*data.shape[0]
        n_min = float("inf")
        n_max = -float("inf")
        for idx in range(data.shape[0]):
            self.distribution[idx] = np.linalg.norm(data[idx]-centroid)
            n_max = self.distribution[idx] if self.distribution[idx] > n_max else n_max
            n_min = self.distribution[idx] if self.distribution[idx] < n_min else n_min
        n_range = n_max - n_min
        self.dist = [np.where(np.logical_and(
                            np.less_equal(self.distribution,n_min + (n_range*self.dist_keys[didx])),
                            np.greater_equal(self.distribution,n_min + (n_range*self.dist_keys[didx-1]))))[0].shape[0] for didx in range(self.nBins+1)]

        # Now that distribution is set up, we need to obtain the Data characteristics, namely the concentration of points...
        #self.max_peak_key = self.distribution.dist.index(max(self.distribution.dist))
        self.delta_low_index, self.delta_high_index, _ = self.getSubArray(self.dist, self.nBins, int(self.alpha*data.shape[0]))
        self.attrs["delta_low"] = self.dist_keys[self.delta_low_index] - (1./self.nBins)
        self.attrs["delta_high"] = self.dist_keys[self.delta_high_index]



    # https://www.geeksforgeeks.org/subarray-whose-absolute-sum-is-closest-to-k/
    # Improve with nlogn solution at https://www.geeksforgeeks.org/subarray-whose-sum-is-closest-to-k/
    # TODO
    def getSubArray(self,arr, n, K): 
        currSum = 0
        prevDif = 0
        currDif = 0
        result = [-1, -1, abs(K-abs(currSum))] 
        resultTmp = result 
        i = 0
        j = 0
        while(i<= j and j<n): 
            currSum += arr[j] 
            prevDif = currDif 
            currDif = K - abs(currSum) 
            if(currDif <= 0): 
                if abs(currDif) < abs(prevDif): 
                    resultTmp = [i, j, currDif] 
                else: 
                    resultTmp = [i, j-1, prevDif] 
                currSum -= (arr[i]+arr[j])                 
                i += 1
            else: 
                resultTmp = [i, j, currDif] 
                j += 1                
            if(abs(resultTmp[2]) < abs(result[2])): 
                result = resultTmp 
        return result 

    def get(self,_key):
        return self.attrs[_key]