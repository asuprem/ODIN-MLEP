import mlep.text.DataCharacteristics.OnlineSimilarityDistribution as OnlineSimilarityDistribution
import mlep.metrics.TextMetrics as TextMetrics
class CosineSimilarityDataCharacteristics:
    def __init__(self, nBins=40, alpha = 0.6):
        self.nBins = nBins
        self.distribution = OnlineSimilarityDistribution.OnlineSimilarityDistribution(nBins)
        self.alpha = alpha


    def buildDistribution(self,centroid,data):
        """ Build the data distribution given a data set and its centroid.

        Uses cosine_similary metric to build the characteristics 'map' of a text-based dataset

        """

        self.centroid = centroid
        for _row in data:
            self.distribution.update(TextMetrics.inverted_cosine_similarity(centroid,_row))

        # Now that distribution is set up, we need to obtain the Data characteristics, namely the concentration of points...
        #self.max_peak_key = self.distribution.dist.index(max(self.distribution.dist))
        self.delta_low_index, self.delta_high_index, _ = self.getSubArray(self.distribution.dist, self.nBins, int(self.alpha*data.shape[0]))
        self.delta_low = self.distribution.dist_keys[self.delta_low_index] - (1./self.nBins)
        self.delta_high = self.distribution.dist_keys[self.delta_high_index]



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
        if _key == "centroid":
            return self.centroid
        else:
            raise NotImplementedError()