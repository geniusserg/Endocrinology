import pandas as pd
import json

class DefaultData:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.dataset = pd.read_excel(self.dataset_path)
        with open("columns.json", "r") as f:
            self.config = json.load(f)

    def preprocess(self):
        dataset = self.dataset.copy()
        parameters_truncated = self.config.parameters_truncated
        dataset["Лечение"] = dataset["ID"].apply(lambda x: x[:3])
        dataset.loc[dataset["Лечение"] == "VBD", "Лечение"] = "STD"
        dataset = dataset.drop((dataset[dataset["Лечение"] == "Red"]).index)
        dataset = dataset.drop("ID", axis=1)

        dataset.columns = list(map(lambda x: x.strip(), dataset.columns))
        dataset.columns = list(map(lambda x: " ".join(x.split()), dataset.columns))
        dataset.columns = list(
            map(lambda i: parameters_truncated[i] if i in parameters_truncated else i, dataset.columns))

        dataset["% потери веса 3 мес"] = ((dataset["Вес 0 мес"] - dataset["Вес 3 мес"]) / dataset["Вес 0 мес"]) * 100
        dataset["% потери веса 6 мес"] = ((dataset["Вес 0 мес"] - dataset["Вес 6 мес"]) / dataset["Вес 0 мес"]) * 100
        dataset[dataset["ИМТ 0 мес"].isna()]["ИМТ 0 мес"] = (dataset[dataset["ИМТ 0 мес"].isna()]["Вес 0 мес"]
                                                             / dataset[dataset["ИМТ 0 мес"].isna()][
                                                                 "Рост"] ** 2).round()

        dataset = dataset.loc[~pd.isna(dataset['Вес 3 мес']) | ~pd.isna(dataset['Вес 6 мес'])]
        errors = ["101,,3", "1,87,54", "0,,38"]
        cols = dataset.columns[(dataset.isin(errors)).any()]
        for col in cols:
            dataset.loc[dataset[col] == "101,,3", col] = 101.3
            dataset.loc[dataset[col] == "1,87,54", col] = 1.8754
            dataset.loc[dataset[col] == "0,,38", col] = 0.38
        dataset["Мочевая Кислота"] = dataset["Мочевая Кислота"].astype(float)
        return dataset

    def get_X_y(self, medicine, params=None):
        dataset = self.dataset.copy()

        dataset = dataset[dataset["Лечение"] == medicine].drop("Лечение", 1)

        targetcol = "% потери веса 3 мес"
        targetcol_6month = "% потери веса 6 мес"
        y_3 = dataset[targetcol]
        treshold = 5
        target = (y_3.astype(int) > treshold).astype(int).reset_index(drop=True).to_numpy()

        y = target
        X = dataset

        if params != None:
            X = dataset[params]

        X = X.loc[:, ~X.isna().all()]
        X = X.reset_index(drop=True)

        for j in [self.config.targets + self.config.initial_columns]:
            if (j in X.columns):
                X = X.drop(j, axis=1)


        return X, y

