


class DataEncoder():
    """ Super class for Data Encoders. They must implement methods presented here """

    def __init__(self,):
        """ Initialize Encoder; usually do nothing"""
        pass

    def setup(self,**args):
        """ Set up the encoder; Args, if needed"""
        pass

    def encode(self, data):
        """ Encode given data into some format """
        
        pass
        encodedData = data
        return encodedData


    def batchEncode(self, data):
        """ Batch encode data into some format """
        pass

    def failCondition(self,**failArgs):
        """ MLEPServer will call this function if encoder setup fails 
        
        Return value must be a boolean. 
            --  True: Encoder is properly set up now. This is useful in case of encoders requiring data downloads, etc. 
                It *is* possible to add this setup in __init__ itself. This is just another option (is this even a good idea - not really a s/w dev here, so...)

            -- False: Encoder errors could not be fixed. This is a bad encoder and should be discarded
        """
        return False