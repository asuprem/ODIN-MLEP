__metaclass__ = type



class BaseLearningModel:
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
        self.track_drift = False

    def fit(self, X, y):
        """Fit the statistical learning model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self._fit(X,y)
        
    def _fit(self,X,y):
        """ Internal function to fit statistical model

        This is the one that should be modified for derived classes

        """
        self._model.fit(X, y)

    def update(self, X, y):
        """Update the statistical learning model to the training data.
        X -- [array of shape (n_samples, n_features)] Training data.
        y -- [array of shape (n_samples)] Target values for the training data.
        """
        self._update(X,y)

    def _update(self,X,y):
        """ Internal function to update the statistical model

        This is the one that should be modified for derived classes

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
        precision, recall, score = self._update_and_test(X_train, y_train, split = split, X_test = X_test, y_test = y_test)

        return precision, recall, score

    def _update_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None):
        """ Internal update_and_test method

        This is the one that should be modified for derived classes

        """
        if X_test is None and y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
    
        self.update(X_train, y_train)
        precision, recall, score = self._precision_recall_fscore(X_test, y_test)
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
        precision, recall, score = self._fit_and_test(X_train, y_train, split = split, X_test = X_test, y_test = y_test)
        return precision, recall, score

    def _fit_and_test(self, X_train, y_train, split = 0.7, X_test = None, y_test = None):
        """ Internal function to fit the statistical model.

        This is the one that should be modified for derived classes

        """
        if X_test is None and y_test is None:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=1.0-split, random_state = 42, shuffle=True, stratify=y_train)
    
        self.fit(X_train, y_train)
        precision, recall, score = self._precision_recall_fscore(X_test, y_test)
        return precision, recall, score

    def _precision_recall_fscore(self, X, y):
        """Return a 3-tuple where the first element is the precision of the model, the second is the
        recall, and the third is the F-measure.
        For statistical learning models, the test data is represented by an array of dimension
        (n_samples, n_features);
        
        X -- [array] Test data.
        y -- [array] Target values for the test data.
        """
        from sklearn.metrics import precision_recall_fscore_support
        y_pred = self.predict(X, mode="test")
        prs_ = tuple(precision_recall_fscore_support(y, y_pred, average="weighted")[0:3])
        return prs_[0], prs_[1], prs_[2]

    def predict(self, X_sample, mode = "predict", y_label = None):
        """Return predicted labels for the test data.

        Given a data point, predict will perform one of three things based on the value of mode:
            - predict/test -- Return the model's prediction of the label of a given sample X_sample
            - implicit --   Perform prediction and return the mode's prediction of label. In addition, store a history of the model's confidence and set internal drift detector
                            Then, calling model's driftDetected() method next time should return True if drift was detected
            
            - explicit --   Perform prediction. Detect drift based on predicted value, confidence, and actual value. Update internal drift detector

        
        Args:
            X_sample: [array of shape (n_samples, n_features)] Test data.
            mode:   "predict" -- for standard prediction. This includes no drift tracking
                    "implicit" -- for implicit drift tracking. This is for unlabeled examples
                    "explicit" -- for explicit drift tracking. This is for labeled examples
                    "test" -- internal-use. No drift tracking is performed. Used during testing/evaluating model post training or update.
        """
        
        
        if mode == "predict" or mode == "test":
            prediction = self._predict(X_sample = X_sample)
        elif mode == "implicit":
            prediction = self._evaluate_implicit_drift(X_sample)
        elif mode == "explicit":
            prediction = self._evaluate_explicit_drift(X_sample, y_label)
        else:
            raise NotImplementedError()
        return prediction


    def _predict(self,X_sample):
        """ Internal function to perform prediction.

        This is the function that should be modified for derived classes
        
        """
        try:
            return self._model.predict(X_sample)
        except ValueError:
            return self._model.predict(X_sample.reshape(1,-1))

    def _evaluate_implicit_drift(self,X_sample):
        """ This evaluates a model's implicit drift """
        
        raise NotImplementedError()

    def _evaluate_explicit_drift(self,X_sample, y_label):
        """ This evaluates a model's explicit drift using y_label """
        
        raise NotImplementedError()


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

    def trackDrift(self,_track=None):
        """ 
        Set/get whether drift is being tracked in a model

        Args:
            _track: Boolean. True/False for set. None/empty for get

        Returns:
            Bool -- content of self.track_drift
        
        """

        if _track is not None:
            self.track_drift = _track
        return self.track_drift

            
    
    