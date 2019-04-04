__metaclass__ = type



class LearningModel:
    """Abstract learning model."""

    def __init__(self, model):
        """Initialize a learning model.
        model -- [object] Learning model.
        """
        self._model = model

    def fit(self, X, y):
        """Fit the statistical learning model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        pdb.set_trace()
        self._model.fit(X, y)

    def predict(self, X):
        """Return predicted labels for the test data.
        X -- [array of shape (n_samples, n_features)] Test data.
        """
        return self._model.predict(X)

    def precision_recall_fscore(self, X, y):
        """Return a 3-tuple where the first element is the precision of the model, the second is the
        recall, and the third is the F-measure.
        For statistical learning models, the test data is represented by an array of dimension
        (n_samples, n_features);
        
        X -- [array] Test data.
        y -- [array] Target values for the test data.
        """
        y_pred = self.predict(X)
        return tuple(precision_recall_fscore_support(y, y_pred, average="weighted")[0:3])