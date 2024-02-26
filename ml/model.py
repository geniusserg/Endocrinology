import pandas as pd
import pickle
import shap
from matplotlib import pyplot as plt

class Model():
    def __init__(self, model_path, explainer_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(explainer_path, "rb") as f:
            self.explainer = pickle.load(f)

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
        shapplot = shap.plots.force(shap_values)
        return shapplot

    def partial_explain(self, feature_a, feature_b=None):
        shap_values = self.explainer(self.explainer.X)
        shap.dependence_plot(
            feature_a,
            shap_values,
            self.explainer.X,
            show=False,
            interaction_index=feature_b
        )
        plt.tight_layout()
        plt.savefig("partial_shap_plot.jpg")
