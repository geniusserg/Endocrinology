import pandas as pd
import pickle
import shap

class Model():
    def __init__(self, model_path, explainer_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(explainer_path, "rb") as f:
            self.explainer = pickle.load(f)

    def predict(self, data):
        data = pd.DataFrame([data]).astype(float)
        y_pred = self.model.predict(data)[0]
        pred_prob = self.model.predict_proba(data)[0][y_pred]
        return y_pred, pred_prob

    def explain(self, sample):
        if (isinstance(sample, dict)):
            sample = pd.DataFrame(sample, index=[0])
        shap_values = self.explainer(sample)
        shapplot = shap.plots.force(shap_values)
        return shapplot