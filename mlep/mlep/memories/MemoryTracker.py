

class MemoryTracker:
    """ This is a MemoryTracker class to track data during classification 
    
    Args:
        memory_mode -- "default": data_set_class is PseudoJsonTweets. Placeholder for future
    """
    def __init__(self,memory_mode="default"):
        pass
        self.MEMORY_TRACKER= {}
        self.MEMORY_MODE = memory_mode
        self.CLASSIFY_MODE = "binary"
    
    def addNewMemory(self,memory_name, memory_store="local", memory_path = None):
        """ 
        Adds a new memory

        Args:
            memory_name -- Name of the memory
            memory_store -- "local": memory will be saved to disk. "in-memory" -- memory will be in-memory. Not Implemented
            memory_path -- folder where memory will be stored and loaded from is memory_store is "local"

        Raises:
            ValueError
        """
        import os
        # memory_path -- ./.MLEPServer/data/
        if memory_name in self.MEMORY_TRACKER:
            raise ValueError("memory_name: %s    already exists in this memory" % memory_name)
        if memory_store == "local" and memory_path is None:
            raise ValueError("Must provide memory_path if using 'local' memory_store.")
        
        if self.MEMORY_MODE == "default":
            if memory_store == "local":
                from mlep.data_model.BatchedLocal import BatchedLocal
                from mlep.data_set.PseudoJsonTweets import PseudoJsonTweets
                data_source = memory_name + "_memory.json"
                data_source_path = os.path.join(memory_path, data_source)
                self.MEMORY_TRACKER[memory_name] = BatchedLocal(data_source=data_source_path, data_mode="single", data_set_class=PseudoJsonTweets)
                self.MEMORY_TRACKER[memory_name].open(mode="a")
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def addToMemory(self,memory_name, data):
        """ 
        Add data to memory 
        
        Args:
            memory_name -- name of memory to which data will be added
            data -- data item to add
            
        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """
        # TODO add integrity check -- is data of the same type as memory's datatype???
        self.MEMORY_TRACKER[memory_name].write(data)
    
    def clearMemory(self,memory_name):
        """ 
        Clear memory

        Args:
            memory_name -- name of memory to clear

        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """
        self.MEMORY_TRACKER[memory_name].clear()

    def hasSamples(self,memory_name):
        """ 
        Return whether memory has samples.

        Args:
            memory_name -- name of memory

        Returns:
            bool: True is memory has items. False if memory has no items

        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """

        return self.MEMORY_TRACKER[memory_name].hasSamples()

    def memorySize(self,memory_name):
        """ 
        Return total number of samples in memory.

        Args:
            memory_name -- name of memory

        Returns:
            INT -- number of samples in memory

        Raises:
            KeyError is memory_name is not in MEMORY_TRACKER (implicit)
        """

        return self.MEMORY_TRACKER[memory_name].memorySize()


