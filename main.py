## Application for obesity treatment 
## Sergey Danilov
from ml.model import Model
import json
from flask import Flask, render_template, request, send_from_directory
import os 
import shap 

app = Flask(__name__,  template_folder='templates')

config: dict = None
model: Model = None


def load_global_config(config_json_path = "config.json", model_snapshots_dir = "model_snapshots"):
    global config, model
    config = json.load(open(config_json_path, "r", encoding="utf-8"))
    model_path = os.path.join(model_snapshots_dir, config["model_path"])
    explainer_path = os.path.join(model_snapshots_dir, config["explainer_path"])
    model = Model(model_path = model_path, explainer_path=explainer_path)

    # small test on Model work
    sample_data = config["sample_data"]
    model.predict(sample_data)


def run_model(input_data):
    return model.predict(data=input_data)


def get_explanation(input_data):
    return model.explain(input_data)


@app.route('/', methods=['GET'])
def index():
    sample_data = config["sample_data"]
    fields = [{"name": i, "placeholder": "Введите значение", "default": sample_data[i], "value": sample_data[i] } for i in sample_data] 
    print(fields)
    return render_template('index.html', fields = fields)


@app.route('/', methods=['POST'])
def submit():
    data = request.form
    sample_data = config["sample_data"]
    fields = [{"name": i, "placeholder": "Введите значение", "default": sample_data[i], "value": sample_data[i] } for i in sample_data] 
    for field in fields:
        field['value'] = data.get(field['name'], '')
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
    return render_template('index.html', fields = fields, result = result, confidence = round(confidence*100, 2), shap_plot = shap_frame)


if __name__ == '__main__':
    # Load some configurations
    load_global_config(
        "config.json",
        "model_snapshots"
    )

    # Start application
    app.run(debug=True)