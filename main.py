## Application for obesity treatment 
## Sergey Danilov
from ml.model import Model
import json
from flask import Flask, render_template, request, send_from_directory, url_for
import os 
import shap 
import pandas as pd
import shutil

app = Flask(__name__,  template_folder='templates')

########
# Model
########

config: dict = None
model: Model = None

def load_global_config(config_json_path = "config.json", model_snapshots_dir = "model_snapshots"):
    global config, model
    config = json.load(open(config_json_path, "r", encoding="utf-8"))

    for model_configuration in ["model_3month", "model_6month"]:
        snapshot_name = config[model_configuration]
        classifier_path = os.path.join(model_snapshots_dir, snapshot_name, "model.pkl")
        explainer_path = os.path.join(model_snapshots_dir, snapshot_name, "explainer.pkl")
        config[model_configuration] = Model(model_path = classifier_path, explainer_path = explainer_path)

    switch_mode("model_3month")

    # small test on Model work
    sample_data = config["sample_data"]
    model.predict(sample_data)

    config["last_data"] = None
    config["last_shap_plot"] = None
    config["last_result"] = (None, None)

def switch_mode(mode):
    global model, config
    if (("mode" in config) and (config["mode"] == mode)):
        return
    config["mode"] = mode
    model = config[config["mode"]]
    model.overall_shap_plot()
    config["features"] = model.get_features()

def run_model(input_data):
    return model.predict(input_data)

def get_explanation(input_data):
    return model.explain(input_data)

def get_partial(feature_a, feature_b=None):
    return model.partial_explain(feature_a, feature_b)

######
# GUI 
######

# Render application (TODO: make common to all)
def render_welcome_page():
    data = config["sample_data"] if config["last_data"] is None else config["last_data"]
    fields = [{"name": i, "placeholder": "Введите значение", "value": data[i] if i in data else config["sample_data"][i] } for i in config["features"]]
    config["last_result"] = (None, None)
    return render_template('index.html', fields = fields, features = config["features"], mode = config["mode"])

@app.route('/', methods=['GET'])
def index():
    if "mode" in request.args:
        switch_mode(request.args["mode"])
    return render_welcome_page()

# Run model and expalin solution
@app.route('/predict', endpoint="predict", methods=['POST'])
def predict():
    data = request.form
    fields = [{"name": i, "placeholder": "Введите значение", "value": data.get(i, '') } for i in config["features"]]
    input_data = {}
    for p in data:
        if data[p] != '':
            try:
                input_data[p] = float(data[p])
            except:
                input_data[p] = None
        else:
            input_data[p] = None
    result, confidence = run_model(input_data)
    get_explanation(input_data)
    config["last_data"] = {i["name"]: i["value"] for i in fields}
    config["last_result"] = (result, confidence)
    return render_template('index.html', fields = fields, result = result, confidence = confidence, features = config["features"], mode = config["mode"])


# Partial explain model
@app.route('/explain', endpoint="explain", methods=['POST'])
def explain():
    data = request.form
    feature_a = data.get("feature1", None)
    feature_b = data.get("feature2", None)
    if (feature_a in config["features"]):
        get_partial(feature_a, feature_b if feature_b in config["features"] else None)
    data = config["sample_data"] if config["last_data"] is None else config["last_data"]
    fields = [{"name": i, "placeholder": "Введите значение", "value": data[i] if i in data else config["sample_data"][i] } for i in config["features"]]
    return render_template('index.html', fields = fields, 
            result = config["last_result"][0],
            confidence = config["last_result"][1],
            features = config["features"],
            image_path = True,
            mode = config["mode"]
    )


if __name__ == '__main__':
    # Load configurations
    load_global_config(
        "config.json",
        "model_snapshots"
    )

    # Start application
    app.run(debug=True)