import json
from ml.modelXGBoostClassifier import XGBClassifier
class State():
    def __init__(self, ml_predictor):
        self.input_params = {}
        with open("data_preprocessing/columns.json", "r") as f:
            self.config = json.load(f)
        self.model = ml_predictor
        self.data = {}

    def predict(self):
        self.model.predict(self.data)


