
from LearningModel import LearningModel


class sklearnLogReg(LearningModel):
    """Logistic regression learning model wrapper."""

    def __init__(self):
        """Initialize a logistic regression learning model wrapper."""
        from sklearn.linear_model import LogisticRegression
        super(sklearnLogReg,self).__init__(LogisticRegression(solver="lbfgs", max_iter=100000))

    