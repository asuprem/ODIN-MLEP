
from LearningModel import LearningModel


class sklearnRandomForest(LearningModel):
    """Decision Tree learning model wrapper."""

    def __init__(self):
        """Initialize a Decison Tree learning model."""
        from sklearn.ensemble import RandomForestClassifier
        super(sklearnRandomForest,self).__init__(RandomForestClassifier(max_depth=10))

    
    def isUpdatable(self):
        """ Decision Tree is not Updatable """
        return False
    