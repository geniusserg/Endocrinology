from datetime import datetime
import shap
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import json
from xgboost import XGBClassifier
import numpy as np

class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = pd.read_excel(self.dataset_path)
        with open("data_preprocessing/columns.json", "r", encoding="utf-8") as f:
            self.config = json.load(f)
        

    def preprocess(self, agroup_params_only=True):
        dataset = self.dataset.copy()
        parameters_truncated = self.config["parameters_truncated"]
        columns_list = self.config["all_parameters"]

        dataset.columns = list(map(lambda x: x.strip(), dataset.columns))
        dataset.columns = list(map(lambda x: " ".join(x.split()), dataset.columns))
        dataset = dataset[columns_list]
        
        dataset["Лечение"] = dataset["ID"].apply(lambda x: x[:3])
        dataset.loc[dataset["Лечение"] == "VBD", "Лечение"] = "STD"
        dataset = dataset.drop((dataset[dataset["Лечение"] == "Red"]).index)
        dataset = dataset.drop("ID", axis=1)

        dataset.columns = list(
                        map(lambda i: parameters_truncated[i] if i in parameters_truncated else i, dataset.columns))

        if (agroup_params_only):
            agroup_params = self.config["model_parameters"]
            dataset = dataset[agroup_params]
            
        dataset["% потери веса 3 мес"] = ((dataset["Вес 0 мес"] - dataset["Вес 3 мес"]) / dataset["Вес 0 мес"]) * 100
        dataset["% потери веса 6 мес"] = ((dataset["Вес 0 мес"] - dataset["Вес 6 мес"]) / dataset["Вес 0 мес"]) * 100
        
        dataset[dataset["ИМТ 0 мес"].isna()]["ИМТ 0 мес"] = (dataset[dataset["ИМТ 0 мес"].isna()]["Вес 0 мес"]
                                                             / dataset[dataset["ИМТ 0 мес"].isna()][
                                                                 "Рост"] ** 2).round()
        
        dataset[dataset["ИМТ 3 мес"].isna()]["ИМТ 3 мес"] = (dataset[dataset["ИМТ 3 мес"].isna()]["Вес 3 мес"]
                                                     / dataset[dataset["ИМТ 3 мес"].isna()][
                                                         "Рост"] ** 2).round()
        
        dataset[dataset["ИМТ 6 мес"].isna()]["ИМТ 6 мес"] = (dataset[dataset["ИМТ 6 мес"].isna()]["Вес 6 мес"]
                                                             / dataset[dataset["ИМТ 6 мес"].isna()][
                                                                 "Рост"] ** 2).round()
        
        dataset = dataset.loc[~pd.isna(dataset['Вес 3 мес']) | ~pd.isna(dataset['Вес 6 мес'])]
        errors = ["101,,3", "1,87,54", "0,,38"]
        cols = dataset.columns[(dataset.isin(errors)).any()]
        for col in cols:
            dataset.loc[dataset[col] == "101,,3", col] = 101.3
            dataset.loc[dataset[col] == "1,87,54", col] = 1.8754
            dataset.loc[dataset[col] == "0,,38", col] = 0.38
        dataset["Мочевая Кислота"] = dataset["Мочевая Кислота"].astype(float)

        # выбросы
        dataset = dataset.drop(dataset[(dataset["Лечение"] == "SIB") & (dataset["% потери веса 3 мес"] > 15)].index)
        dataset = dataset.drop(dataset[(dataset["Лечение"] == "SIB") & (dataset["% потери веса 6 мес"] > 15)].index)
        dataset.loc[(dataset["Лечение"] == "SIB") & (dataset["% потери веса 3 мес"] > 5) & (dataset["Возраст"] > 50), "% потери веса 3 мес"] = 4

        self.dataset = dataset
        
    
    def get_X_y(self, medicine, treshold=5, params=None, target_type="A"):
        dataset = self.dataset.copy()
        
        dataset = dataset[dataset["Лечение"] == medicine].drop("Лечение", axis=1)
        
        X = dataset

        if (target_type=="A"):
            y_target = dataset["% потери веса 3 мес"]
        elif (target_type=="B"):
            y_target = dataset["% потери веса 6 мес"]
        elif (target_type=="C"):
            targetcol = ["% потери веса 3 мес", "% потери веса 6 мес"]
            y_target = dataset[targetcol].max(numeric_only = True, axis=1)
        elif (target_type=="both"):
            y_target = dataset["% потери веса 6 мес"]
        else:
            raise Exception("type undiefined")
        
        y_target = y_target[~np.isnan(y_target)]
        X = X.loc[y_target.index]

        y = (y_target >= treshold).astype(int).reset_index(drop=True).to_numpy()

        if (params is None):
            for j in self.config["targets"]:
                if (j in X.columns):
                    X = X.drop(j, axis=1)
        else:
            X = X[params]

        X = X.reset_index(drop=True).astype(float)
        return X, y


def save_experiment(model, X, y, experiment_name, snapshot_folder = "model_snapshots"):
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%d_%m_%H")
    if (not os.path.exists(f"{snapshot_folder}/{experiment_name}_{formatted_datetime}")):
        os.mkdir(f"{snapshot_folder}/{experiment_name}_{formatted_datetime}")
    model_path = f"{snapshot_folder}/{experiment_name}_{formatted_datetime}/model.pkl"
    expaliner_path = f"{snapshot_folder}/{experiment_name}_{formatted_datetime}/explainer.pkl"
    plot_path = f"{snapshot_folder}/{experiment_name}_{formatted_datetime}/shap_overall.jpg"
    data_path = f"{snapshot_folder}/{experiment_name}_{formatted_datetime}/data.csv"

    model.fit(X, y)
    explainer = shap.TreeExplainer(model, X)
    explainer.X = X
    explainer.y = y
    pickle.dump(model, open(model_path, "wb"))
    pickle.dump(explainer, open(expaliner_path, "wb"))
    data = pd.concat([X, pd.Series(y)], axis=1).rename(columns={0: "success"})
    data.to_csv(data_path)

    shap.summary_plot(explainer(X)[:, :], X, show=False)
    plt.tight_layout()
    plt.savefig(plot_path)
    return f"{snapshot_folder}/{experiment_name}_{formatted_datetime}"



dt = Dataset(dataset_path="data/dataset_almazov_november.xlsx"); dt.preprocess(agroup_params_only=False)


# 3 months
params =  ["Возраст", "ИМТ 0 мес", "СРБ", "Лептинрезистентность", "Глюкоза", 'СКФ', "ДАД", "OXC"]
X, y = dt.get_X_y("SIB", 5, params=params, target_type="A")
model = XGBClassifier()
model.fit(X, y)
experiment_name = "xgboost_agroup_3month"
exp_name = save_experiment(model, X, y, experiment_name)
print(f"Model saved: {exp_name}")

# 6 months
params = ["Возраст", "ИМТ 3 мес", "СРБ", "Лептинрезистентность", "Глюкоза", 'СКФ', "ДАД", "% потери веса 3 мес"]
X, y = dt.get_X_y("SIB", 7, params=params, target_type="both")
model = XGBClassifier()
model.fit(X, y)
experiment_name = "xgboost_agroup_6month"
exp_name = save_experiment(model, X, y, experiment_name)
print(f"Model saved: {exp_name}")

# 3 months

params = ["Возраст", "ИМТ 0 мес", "СРБ", "Лептинрезистентность", "Глюкоза", 'СКФ', "ДАД", "OXC", 'ГПП 1 нг/мл 0 мес', 'ГИП (пг/мл) 0 мес', 'Грелин (нг/мл) 0 мес', 'miR142 (ПЛАЗМА) 0 мес', 'Проколлаген 1 типа нг/мл (183-244) 0 мес', 'Проколлаген 3 типа пг/мл (11178-36957) 0 мес', 'MMP-9 нг/мл (450-1351) 0 мес', 'sST2 нг/мл (15,15-26,86) 0 мес']
X, y = dt.get_X_y("SIB", 5, params=params, target_type="A")
model = XGBClassifier()
model.fit(X, y)
experiment_name = "xgboost_bgroup_3month"
exp_name = save_experiment(model, X, y, experiment_name)
print(f"Model saved: {exp_name}")
# 6 months

params = ["Возраст", "ИМТ 3 мес", "СРБ", "Лептинрезистентность", "Глюкоза", 'СКФ', "ДАД", "% потери веса 3 мес", 'ГПП 1 нг/мл 0 мес', 'ГИП (пг/мл) 0 мес', 'miR142 (ПЛАЗМА) 0 мес']
X, y = dt.get_X_y("SIB", 7, params=params, target_type="both")
model = XGBClassifier()
model.fit(X, y)
experiment_name = "xgboost_bgroup_6month"
exp_name = save_experiment(model, X, y, experiment_name)
print(f"Model saved: {exp_name}")
