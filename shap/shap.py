from ml.model import Model
import shap


class SHAP():
    def __init__(self, model: Model):
        self.model = model.model
        self.explainer = None

    def create_explainer(self, X):
        self.explainer = shap.TreeExplainer(self.model, X)
        self.shap_values = self.explainer(X)
        return self.shap_values

    def shap_summary_plot(self, model, X):
        shap.summary_plot(self.shap_values[:, :], X)

# TODO: how to plot to image?