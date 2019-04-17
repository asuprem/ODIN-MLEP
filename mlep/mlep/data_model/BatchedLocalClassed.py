import mlep.data_model.BatchedLocal

class BatchedLocalClassed(mlep.data_model.BatchedLocal.BatchedLocal):
    """ BatchedLocal model loads the batchedlocal file"""
    
    def __init__(self, data_source=None, data_mode=None, num_samples="all", data_set_class=None, classification_mode="binary", classes=[0,1]):
        """

        data_source -- path of the Local Batched Data File

        num_samples -- how many samples to load at a time...
                -- also not implemented. For now we load all examples in memory
                -- TODO add option for lazy loading vs preloading
        
        data_mode -- [collected/sequential]
            - "single" -- all samples are in a single file - the one linked in data_source
            - "sequential" - samples are spread out across multiple files. Probably will use some form of wildcarding for this. TBD.
            - "split" - each sample is a file. Each sample must be in their corresponding label folder (or something) - TBD. Sort of like keras data generator


            We only handle collected so far. Batched also means all data is loaded into memory

        data_set_class -- Class file (from DataSet) that encapsulates each example.

        examples are delimited by newlines
        """
        self.class_data={}
        self.class_statistics={}
        self.classes=classes
        for _class in classes:
            self.class_data[_class] = []
            self.class_statistics[_class] = 0
        

        self.data_mode = data_mode
        self.data_source = data_source
        self.data_set_class = data_set_class



        # so self.data is a list of [data_set_class(), data_set_class()...]  
    def load_by_class(self,):
        if self.data_mode == "single":
            with open(self.data_source,"r") as data_source_file:
                for line in data_source_file:
                    self.data.append(self.data_set_class(line))
                    assert(self.data[-1].getLabel() in self.classes)
                    self.classes[self.data[-1].getLabel()].append(self.data[-1])
                    self.class_statistics[self.data[-1].getLabel()] += 1
    
    def all_class_sizes(self,):
        return sum([self.class_statistics[item] for item in self.class_statistics])
    def class_size(self,_class):
        return self.class_statistics[_class]

    
    
    def getData(self,_class):
        return [dataItem.getData() for dataItem in self.data]

    def getLabels(self,_class):
        return [dataItem.getLabel() for dataItem in self.data]

    def getObjects(self,_class):
        return self.data
    def augment(self,data,labels):
        pass


    def getNextBatchData(self,):
        raise NotImplementedError()

    def getNextBatchLabels(self,):
        raise NotImplementedError()





