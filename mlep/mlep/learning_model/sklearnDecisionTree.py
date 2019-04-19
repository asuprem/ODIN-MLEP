
import mlep.learning_model.LearningModel

class sklearnDecisionTree(mlep.learning_model.LearningModel.LearningModel):
    """Decision Tree learning model wrapper."""

    def __init__(self):
        """Initialize a Decison Tree learning model."""
        from sklearn.tree import DecisionTreeClassifier
        super(sklearnDecisionTree,self).__init__(DecisionTreeClassifier(max_depth=10))

    
    def isUpdatable(self):
        """ Decision Tree is not Updatable """
        return False
    