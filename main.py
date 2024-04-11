## Application for obesity treatment 
## Sergey Danilov
from ml.model import Model
import json
import os 
from flask import Flask, render_template, request

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

    config["last_data"] = None
    config["last_shap_plot"] = None
    config["last_result"] = (None, None)
    config["mode"] = "model_3month"
    config["extra"] = False
    model = config["model_agroup_3month"]


def run_model(input_data):
    return model.predict(input_data)

def get_explanation(input_data):
    return model.explain(input_data)

def get_partial(feature_a, feature_b=None):
    return model.partial_explain(feature_a, feature_b)

def transform_input_data(input_data):
    if (("Лептин" in input_data) and (input_data["Лептин"] is not None)):
        if (("Лептин 1 час" in input_data) and (input_data["Лептин 1 час"] is not None)):
            input_data["Постпрандиальная динамика лептина"] = ((input_data["Лептин 1 час"] - input_data["Лептин"])/ input_data["Лептин"])*100
    if (("ИМТ 3 мес" in input_data) and (input_data["ИМТ 3 мес"] is not None)):
        if (("ИМТ 0 мес" in input_data) and (input_data["ИМТ 0 мес"] is not None)):
            input_data["% потери веса 3 мес"] = ((input_data["ИМТ 0 мес"] - input_data["ИМТ 3 мес"])/(input_data["ИМТ 0 мес"]))*100
            input_data["% потери веса 3 мес"] = round(input_data["% потери веса 3 мес"], 2)
    return input_data


######
# GUI 
######

# Render application 
def render_welcome_page():
    data = {i: None for i in config["features"]}
    if (config["last_data"] is not None): # load saved data or from deafult values from config
        data = {i: config["last_data"][i] if i in config["last_data"] else None for i in config["features"]}
    else:
        data = {i: config["sample_data"][i] if i in config["sample_data"] else None for i in config["features"]}
    # descriptions = {i: config["descriptions"][i] if i in config["descriptions"] else i for i in data}
    fields = [{"name": i, "description": i, "value": data[i]} for i in data]
    confidence = config["last_result"][1] if config["last_result"][1] is not None else None
    result = config["last_result"][0] if config["last_result"][0] is not None else None
    return render_template('index.html', fields = fields, result = result, confidence = confidence, features = list(data.keys()), mode = config["mode"])

@app.route('/', methods=['GET'])
def index():
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
    input_data = transform_input_data(input_data)
    input_data = {i: input_data[i] for i in model.get_features()}
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