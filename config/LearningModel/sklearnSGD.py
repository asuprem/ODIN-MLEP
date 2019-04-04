from LearningModel import LearningModel


class sklearnSGD(LearningModel):
    """SVM learning model wrapper."""

    def __init__(self):
        """Initialize a SVM learning model wrapper."""
        from sklearn.linear_model import SGDClassifier
        super(sklearnSGD,self).__init__(SGDClassifier())