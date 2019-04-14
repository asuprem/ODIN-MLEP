from DataSet import DataSet

class PseudoJson(DataSet):
    """ PseudoJson model parses a psuedojson line and extracts relevant details from it """

    def __init__(self,data, dataKey, labelKey):
        self.raw = data
        self.data = self.raw["dataKey"]
        self.label = self.raw["labelKey"]

    def getData(self,):
        return self.data

    def getLabel(self,):
            return self.label

    def getValue(self,key):
        return self.raw[key]


    