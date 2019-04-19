import mlep.data_model.DataModel

class StreamLocal(mlep.data_model.DataModel.DataModel):
    """ StreamLocal model. Simulates streaming data"""
    
    def __init__(self, data_source=None, data_mode=None, num_samples="all", data_set_class=None):
        """

        data_source -- path of the Local Batched Data File

        num_samples -- how many samples to load at a time...
                -- also not implemented. For now we load all examples in memory
                -- TODO add option for lazy loading vs preloading
        
        data_mode -- [collected/sequential]
            - "single" -- all samples are in a single file - the one linked in data_source
            - "sequential" - samples are spread out across multiple files. Probably will use some form of wildcarding for this. TBD.

            We only handle collected so far. Batched also means all data is loaded into memory

        data_set_class -- Class file (from DataSet) that encapsulates each example.

        examples are delimited by newlines
        """

        # Init function loads data
        # load into a dataframe---?
        self.data=[]
        if data_mode == "single":
            with open(data_source,"r") as data_source_file:
                for line in data_source_file:
                    self.data.append(data_set_class(line))
        self.idx=0

        # so self.data is a list of [data_set_class(), data_set_class()...]  

    """ TODO make this an iterator format. Also learn aboout iteratoro formars :/"""
    def getData(self,):
        return self.data_

    def getLabel(self,):
        return self.label_

    def getObject(self,):
        return self.object_
        
    def next(self,):
        try:
            self.object_ = self.data[self.idx]
        except IndexError:
            return False

        self.data_ = self.object_.getData()
        self.label_ = self.object_.getLabel()
        self.idx+=1
        return True
        
        

    
    
    def getNextBatchData(self,):
        raise NotImplementedError()

    def getNextBatchLabels(self,):
        raise NotImplementedError()





