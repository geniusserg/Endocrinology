## Application for obesity treatment 
## Sergey Danilov


from ui.gui import View
from ml.modelXGBoostClassifier import ModelXGBoostClassifier
import json

config = json.load("config.json")

m = ModelXGBoostClassifier(model_path="model_snapshots/model.pkl")
sample_data = {'OXC': 6.13, 'ДАД': 70.0, 'Возраст': 39.0, 'Лептин (нг/мл) 0 мес': 92.63}
print(m.predict(data=sample_data))

