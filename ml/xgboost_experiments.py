from xgboost import XGBoostClassifier

from data_preprocessing.default_dataset import DefaultData
from model import Model


class ModelXGBoost(Model):
    def __init__(self):
        super().__init__()
        self.model = XGBoostClassifier()


# model for experiments

t = ModelXGBoost()
d = DefaultData(dataset_path="data/Грант НЦМУ - Ожирение 17.09.2023 (2).xlsx")
d.dataset = d.preprocess()

t.run_experiment(d.get_X_y("SIB"))

