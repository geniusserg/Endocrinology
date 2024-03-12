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

    for model_configuration in ["model_agroup_3month", "model_agroup_6month", "model_bgroup_3month", "model_bgroup_6month"]:
        classifier_path = os.path.join(model_snapshots_dir, model_configuration, "model.pkl")
        explainer_path = os.path.join(model_snapshots_dir, model_configuration, "explainer.pkl")
        config[model_configuration] = Model(model_path = classifier_path, explainer_path = explainer_path)

    switch_mode("model_3month")

    config["last_data"] = None
    config["last_shap_plot"] = None
    config["last_result"] = (None, None)

def switch_mode(mode, extra=False):
    global model, config
    config["mode"] = mode
    config["extra"] = extra
    print(mode, extra)
    if ((config["mode"] == "model_3month") and (config["extra"] == False)):
        model = config["model_agroup_3month"]
    if ((config["mode"] == "model_6month") and (config["extra"] == False)):
        model = config["model_agroup_6month"]
    if ((config["mode"] == "model_3month") and (config["extra"] == True)):
        model = config["model_bgroup_3month"]
    if ((config["mode"] == "model_6month") and (config["extra"] == True)):
        model = config["model_bgroup_6month"]
    model.overall_shap_plot()
    config["features"] = model.get_features()
    config["last_data"] = None
    config["last_shap_plot"] = None
    config["last_result"] = (None, None)

def run_model(input_data):
    return model.predict(input_data)

def get_explanation(input_data):
    return model.explain(input_data)

def get_partial(feature_a, feature_b=None):
    return model.partial_explain(feature_a, feature_b)

######
# GUI 
######

# Render application 
def render_welcome_page(image_path=False):
    data = config["sample_data"] if config["last_data"] is None else config["last_data"]
    data = {i: data[i] if i in data else '' for i in config["features"]}
    descriptions = {i: config["descriptions"][i] if i in config["descriptions"] else i for i in data}
    fields = [{"name": i, "description": descriptions[i], "placeholder": "Введите значение", "value": data[i]} for i in config["features"]]
    confidence = config["last_result"][1] if config["last_result"][1] is not None else None
    result = config["last_result"][0] if config["last_result"][0] is not None else None
    checked = "checked" if config["extra"] == True else ""
    return render_template('index.html', fields = fields, result = result, confidence = confidence, features = config["features"], mode = config["mode"], image_path=image_path, checked=checked)

@app.route('/', methods=['GET'])
def index():
    if "mode" in request.args:
        switch_mode(request.args["mode"], "extra" in request.args)
    
    return render_welcome_page()

# Run model and expalin solution
@app.route('/predict', endpoint="predict", methods=['POST'])
def predict():
    data = request.form
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
    config["last_data"] = {i: data.get(i, '')  for i in config["features"]}
    config["last_result"] = (result, round(confidence*100, 2))
    return render_welcome_page()

# Partial explain model
@app.route('/explain', endpoint="explain", methods=['POST'])
def explain():
    data = request.form
    feature_a = data.get("feature1", None)
    feature_b = data.get("feature2", None)
    if (feature_a in config["features"]):
        get_partial(feature_a, feature_b if feature_b in config["features"] else None)
    data = config["sample_data"] if config["last_data"] is None else config["last_data"]
    return render_welcome_page(image_path=True)


if __name__ == '__main__':
    # Load configurations
    load_global_config(
        "config.json",
        "model_snapshots"
    )

    # Start application
    app.run()