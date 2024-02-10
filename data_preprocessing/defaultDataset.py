import pandas as pd
import json


class Dataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = pd.read_excel(self.dataset_path)
        with open("data_preprocessing/columns.json", "r") as f:
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

        self.dataset = dataset

    def get_X_y(self, medicine, treshold=5, params=None):
        dataset = self.dataset.copy()

        dataset = dataset[dataset["Лечение"] == medicine].drop("Лечение", axis=1)

        targetcol = "% потери веса 3 мес"
        y_3 = dataset[targetcol]
        target = (y_3 > treshold).astype(int).reset_index(drop=True).to_numpy()
        y = target
        X = dataset

        if params != None:
            X = dataset[params]
        X = X.loc[:, ~X.isna().all()]
        X = X.reset_index(drop=True)

        to_drop = []

        for j in self.config["targets"] + self.config["initial_columns"]:
            if (j in X.columns):
                X = X.drop(j, axis=1)

        X = X.astype(float)
        return X, y