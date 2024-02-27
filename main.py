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

config: dict = None
model: Model = None


def load_global_config(config_json_path = "config.json", model_snapshots_dir = "model_snapshots"):
    global config, model
    config = json.load(open(config_json_path, "r", encoding="utf-8"))
    snapshot_name = config["model_path"]
    classifier_path = os.path.join(model_snapshots_dir, snapshot_name, "model.pkl")
    explainer_path = os.path.join(model_snapshots_dir, snapshot_name, "explainer.pkl")
    data_path = os.path.join(model_snapshots_dir, snapshot_name, "data.csv")
    shap_overall_plot = os.path.join(model_snapshots_dir, snapshot_name, "shap_overall.jpg")

    model = Model(model_path = classifier_path, explainer_path = explainer_path)
    if (os.path.exists(shap_overall_plot)):
        shutil.copyfile(shap_overall_plot, os.path.join("static", "plot.jpg"))

    # small test on Model work
    sample_data = config["sample_data"]
    model.predict(sample_data)

    config["last_data"] = None


def run_model(input_data):
    return model.predict(input_data)

def get_explanation(input_data):
    return model.explain(input_data)

def get_partial(feature_a, feature_b=None):
    return model.partial_explain(feature_a, feature_b)

@app.route('/', methods=['GET'])
def index():
    if (config["last_data"] is not None):
        fields = config["last_data"] 
    else:
        sample_data = config["sample_data"]
        fields = [{"name": i, "placeholder": "Введите значение", "default": sample_data[i], "value": sample_data[i] } for i in sample_data]
        config["last_data"] = fields
        config["last_shap_plot"] = None
        config["last_result"] = (None, None)
    return render_template('index.html', fields = fields, features = config["features"])


@app.route('/predict', endpoint="predict", methods=['POST'])
def submit():
    data = request.form
    fields = config["last_data"]
    for idx, field in enumerate(config["features"]):
        fields[idx]['value'] = data.get(field, '')
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
    shap_plot = get_explanation(input_data)
    shap_frame = f"<head>{shap.getjs()}</head><body>{shap_plot.html()}</body>"
    config["last_data"] = fields
    config["last_shap_plot"] = shap_frame
    config["last_result"] = (result, confidence)
    return render_template('index.html', fields = fields, result = result, confidence = confidence, shap_plot = shap_frame, features = config["features"])

@app.route('/explain', endpoint="explain", methods=['POST'])
def submit():
    data = request.form
    feature_a = data.get("feature1", config["features"][0])
    feature_b = data.get("feature2", None)
    get_partial(feature_a, feature_b)
    if (os.path.exists(os.path.join("static", "partial_shap_plot.jpg"))):
        print("OK !")
    else:
        print(f'NO {os.path.join("static", "partial_shap_plot.jpg")}')
    return render_template('index.html', fields = config["last_data"], 
            result = config["last_result"][0],
            confidence = config["last_result"][1],
            shap_plot = config["last_shap_plot"],
            features = config["features"]+[None],
            image_path = True
    )


if __name__ == '__main__':
    # Load some configurations
    load_global_config(
        "config.json",
        "model_snapshots"
    )

    # Start application
    app.run(debug=True)