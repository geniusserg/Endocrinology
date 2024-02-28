import pandas as pd
import pickle
import shap
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams, pyplot as plt
rcParams.update({'figure.autolayout': True})

class Model():
    def __init__(self, model_path, explainer_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(explainer_path, "rb") as f:
            self.explainer = pickle.load(f)
        self.dataset_shap_values = self.explainer(self.explainer.X)

    def get_features(self):
        return self.model.feature_names_in_

    def predict(self, data_input):
        data = {k:data_input[k] for k in self.model.feature_names_in_}
        data = pd.DataFrame([data]).astype(float)
        y_pred = self.model.predict(data)[0]
        pred_prob = self.model.predict_proba(data)[0][y_pred]
        print(y_pred, pred_prob)
        return y_pred, pred_prob

    def explain(self, data_input):
        data = {k:data_input[k] for k in self.model.feature_names_in_}
        data = pd.DataFrame(data, index=[0])
        shap_values = self.explainer(data)
        plt.ioff(); fig = plt.figure()
        shap.plots.waterfall(shap_values[0, :], show=False)
        plt.savefig(os.path.join("static", "explain.jpg")); plt.close(fig)
        return "explain.jpg"

    def partial_explain(self, feature_a, feature_b=None):
        plt.ioff(); fig = plt.figure()
        shapplot = shap.dependence_plot(
            feature_a,
            self.dataset_shap_values[:, :].values,
            self.explainer.X,
            interaction_index=feature_b if feature_b is not None else feature_a,
            show=False
        )
        plt.savefig(os.path.join("static", "partial_explain.jpg")); plt.close(fig)
        return shapplot

