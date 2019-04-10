


class EncoderGenerator():
    """ Super class for Encoder Generator. They must implement methods presented here 
    Encoder Generator creates an Encoder for DataEncoder to use
    
    Maybe we can combine them in the future?"""

    def __init__(self,):
        """ Initialize Encoder Generator"""
        pass

    def generate(self, ):
        """ Generate an encoder. EncoderGenerator will use that encoder as the basis. So encoder can be a .bin file in the case of w2v, or perhaps a 
        pandas matrix in the case of ESA??

        MUST be saved in the Sources folder. We can have controls for this later on. 
        """
        
        pass
