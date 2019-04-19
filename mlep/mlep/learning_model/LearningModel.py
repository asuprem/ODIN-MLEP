__metaclass__ = type



class LearningModel:
    """Abstract learning model."""

    def __init__(self, model, mode="binary",classes=[0,1]):
        """Initialize a learning model.
        model -- [object] Learning model.
        mode  -- [str] Mode of learning (binary, multiclass, or regression)
        model -- [int] Number of classes. None for regression
        """
        self._model = model
        self.mode = mode
        self.classes = classes

    def fit(self, X, y):
        """Fit the statistical learning model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self._model.fit(X, y)

    def update(self, X, y):
        """Update the statistical learning model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self._model.partial_fit(X, y, classes=self.classes)

    def update_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None):
        """Update the statistical learning model to the training data and test
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        X_test -- [array of shape (n_samples, n_features)] Testing data.
        y_test -- [array of shape (n_samples)] Target values for the Testing data.

        If X_test and y_test are not provided, split value is used (default 0.7) to shuffle and split X_train and y_train
        """
        # TODO handle weird erros, such as X_test specified, but y_test not specified, etc
        if X_test is None and y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
    
        self.update(X_train, y_train)
        precision, recall, score = self.precision_recall_fscore(X_test, y_test)
        return precision, recall, score

    def fit_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None):
        """Fit the statistical learning model to the training data and test
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        X_test -- [array of shape (n_samples, n_features)] Testing data.
        y_test -- [array of shape (n_samples)] Target values for the Testing data.

        If X_test and y_test are not provided, split value is used (default 0.7) to shuffle and split X_train and y_train
        """
        # TODO handle weird erros, such as X_test specified, but y_test not specified, etc
        if X_test is None and y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
    
        self.fit(X_train, y_train)
        precision, recall, score = self.precision_recall_fscore(X_test, y_test)
        return precision, recall, score

    def predict(self, X):
        """Return predicted labels for the test data.
        X -- [array of shape (n_samples, n_features)] Test data.
        """
        
        try:
            return self._model.predict(X)
        except ValueError:
            return self._model.predict(X.reshape(1,-1))

    def precision_recall_fscore(self, X, y):
        """Return a 3-tuple where the first element is the precision of the model, the second is the
        recall, and the third is the F-measure.
        For statistical learning models, the test data is represented by an array of dimension
        (n_samples, n_features);
        
        X -- [array] Test data.
        y -- [array] Target values for the test data.
        """
        from sklearn.metrics import precision_recall_fscore_support
        y_pred = self.predict(X)
        prs_ = tuple(precision_recall_fscore_support(y, y_pred, average="weighted")[0:3])
        return prs_[0], prs_[1], prs_[2]

    def clone(self, LearningModelToClone):
        """Clone the LearningModelToClone into this model. They must be the same model Type
        
        LearningModelToClone -- [LearningModel] The model to clone
        """
        self._model = LearningModelToClone.clonedModel()

    def clonedModel(self):
        """Return a clone of the current model. This and clone work together
        
        """
        from sklearn.base import clone as sklearnClone
        return sklearnClone(self._model)

    def isUpdatable(self):
        return True