
import mlep.data_set.PseudoJson

class PseudoJsonTweets(mlep.data_set.PseudoJson.PseudoJson):
    """ PseudoJsonTweets model parses a psuedojson line and extracts relevant details from it """

    def __init__(self,data):
        super(PseudoJsonTweets,self).__init__(data, dataKey="text", labelKey="label")


    