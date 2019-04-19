import mlep.learning_model.LearningModel


class sklearnSGD(mlep.learning_model.LearningModel.LearningModel):
    """SVM learning model wrapper."""

    def __init__(self):
        """Initialize a SVM learning model wrapper."""
        from sklearn.linear_model import SGDClassifier
        super(sklearnSGD,self).__init__(SGDClassifier())