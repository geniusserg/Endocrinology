import pandas as pd


class Model():
    def __init__(self):
        self.model = None

    def load_model(self):
        pass

    def predict(self, data):
        data = pd.DataFrame([data])
        y_pred = self.model.predict(data)[0]
        pred_prob = self.model.predict_proba(data)[0][y_pred]
        return y_pred, pred_prob