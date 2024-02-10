from xgboost import XGBClassifier
from ml.model import Model
import pickle

# XGBoostClassifier
class ModelXGBoostClassifier(Model):
    def __init__(self, model_path=None):
        self.model = XGBClassifier()
        if model_path is not None:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)