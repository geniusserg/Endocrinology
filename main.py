## Application for obesity treatment 
## Sergey Danilov
from ml.modelXGBoostClassifier import ModelXGBoostClassifier
import json
from flask import Flask, render_template, request, send_from_directory
import os 
import shap 

app = Flask(__name__,  template_folder='templates')


def run_model(input_data):
    
    return model.predict(data=input_data)

@app.route('/', methods=['GET'])
def index():
    sample_data = config["sample_data"]
    fields = [{"name": i, "placeholder": "Введите значение", "default": sample_data[i], "value": sample_data[i] } for i in sample_data] 
    print(fields)
    return render_template('index.html', fields = fields)

@app.route('/', methods=['POST'])
def submit():
    data = request.form
    # sample_data = config["sample_data"]
    fields = [{"name": i, "placeholder": "Введите значение", "default": sample_data[i], "value": sample_data[i] } for i in sample_data] 
    for field in fields:
        field['value'] = data.get(field['name'], '')
    input_data = {}
    print("DATA: ", data)
    for p in data:
        print(p)
        input_data[p] = float(data[p])
    result, confidence = run_model(input_data)
    return render_template('index.html', fields = fields, result = result, confidence = round(confidence*100, 2))


if __name__ == '__main__':
    # model test
    config = json.load(open("config.json", "r", encoding="utf-8"))
    model_path = os.path.join("model_snapshots", config["model_path"])
    model = ModelXGBoostClassifier(model_path = model_path)

    sample_data = config["sample_data"]
    print(model.predict(sample_data))

    #application
    app.run(debug=True)