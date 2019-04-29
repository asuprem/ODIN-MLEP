import mlep.representations.BaseRepresentation

class ZonedDistribution(mlep.representations.BaseRepresentation.BaseRepresentation):
    def __init__(self, nBins=40, alpha=0.6, distance_metric=None):
        self.distribution = None
        self.alpha = alpha
        self.nBins = nBins

        self.attrs={}
        self.attrs["centroid"] = None
        self.attrs["delta_high"] = None
        self.attrs["delta_low"] = None

    def buildDistribution(self,centroid, data):
        """
        Given centroid, data, and self.distance_metric, use self.distance metric to build a distribution using the data. 

        Then obtain delta_high and delta_low of the range
        """
        pass

    def updateDistribution(self,):
    



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