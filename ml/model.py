import numpy as np

from sklearn.model_selection import train_test_split, learning_curve, LearningCurveDisplay, LeaveOneOut, KFold, GridSearchCV

class Model():
    def __init__(self):
        self.model = None

    def run_experiment(self, X, y):
        cv_loo = LeaveOneOut()
        y_pred = []
        y_true = []
        for idx, (train_index, test_index) in enumerate(cv_loo.split(X)):
            X_train, X_test, y_train, y_test = X.iloc[train_index, :], \
                X.iloc[test_index, :], \
                y[train_index], \
                y[test_index]
            self.model.fit(X_train, y_train)
            y_pred.append(self.model.predict(X_test))
            y_true.append(y_test[0])

        y_pred = np.array(y_pred).squeeze()
        return y_pred
