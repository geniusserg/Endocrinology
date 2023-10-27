import gdown
import pandas as pd
import json

class Dataset():
    def __init__(self):
        self.path = "dataset/ALMAZOV_dataset.xlsx"
        self.columns_json = "dataset/agroup_columns.json"
        self.df: pd.DataFrame = None
        self.download_dataset()
        self.select_columns()
        self.check_dataset()
        print(self.df.describe())

    def download_dataset(self):
        output = self.path
        id = "1mYU2x0NhbIhmuCYAKPSP3Q9345sPlAC1"
        gdown.download(id=id, output=output, quiet=False)
        self.df = pd.read_excel(self.path)
        
    def select_columns(self):
        with open(self.columns_json, "r") as f:
            self.columns_json = json.load(f)
        self.df["Лечение"] = self.df["ID"].apply(lambda x: x[:3] if x[:3] != 'VBD' else 'STD')
        aliases = self.columns_json["aliases"]
        agroup_coluns = self.columns_json["agroup_columns"]
        target_columns = self.columns_json["target_columns"]
        agroup_parameters = list(set(agroup_coluns+target_columns))
        self.df.columns = list(map(lambda x: x.strip(), self.df.columns))
        self.df.columns = list(map(lambda x: " ".join(x.split()), self.df.columns))
        self.df.columns = list(map(lambda i: aliases[i] if i in aliases else i, self.df.columns))
        self.df = self.df[agroup_parameters]

    def check_dataset(self):
        self.df.loc[(self.df["НТГ"]>0) & (self.df["НТГ"]<1) , "НТГ"] = 0

        self.df = self.df.loc[~pd.isna(self.df['Вес 0 мес'])]

        self.df.loc[pd.isna(self.df['% потери веса 3 мес']) & ~pd.isna(self.df['Вес 3 мес']), '% потери веса 3 мес'] = round(
        (self.df.loc[pd.isna(self.df['% потери веса 3 мес']) & ~pd.isna(self.df['Вес 3 мес']), 'Вес 0 мес']
        - self.df.loc[pd.isna(self.df['% потери веса 3 мес']) & ~pd.isna(self.df['Вес 3 мес']), 'Вес 3 мес'])
        / self.df.loc[pd.isna(self.df['% потери веса 3 мес']) & ~pd.isna(self.df['Вес 3 мес']), 'Вес 0 мес'] * 100 , 1
        )

        # self.df.loc[pd.isna(self.df['% потери веса 6 мес']) & ~pd.isna(self.df['Вес 6 мес']), '% потери веса 6 мес'] = round (
        # (self.df.loc[pd.isna(self.df['% потери веса 6 мес']) & ~pd.isna(self.df['Вес 6 мес']), 'Вес 0 мес']
        # - self.df.loc[pd.isna(self.df['% потери веса 6 мес']) & ~pd.isna(self.df['Вес 6 мес']), 'Вес 6 мес'])
        # / self.df.loc[pd.isna(self.df['% потери веса 6 мес']) & ~pd.isna(self.df['Вес 6 мес']), 'Вес 0 мес'] * 100, 1
        # )

        # self.df.loc[~pd.isna(self.df['% потери веса 6 мес']) & pd.isna(self.df['Вес 6 мес']), 'Вес 6 мес'] = round (
        # self.df.loc[~pd.isna(self.df['% потери веса 6 мес']) & pd.isna(self.df['Вес 6 мес']), '% потери веса 6 мес']
        # *self.df.loc[~pd.isna(self.df['% потери веса 6 мес']) & pd.isna(self.df['Вес 6 мес']), 'Вес 0 мес'] /100, 1
        # )

        self.df[self.df["ИМТ 0 мес"].isna()]["ИМТ 0 мес"] = (self.df[self.df["ИМТ 0 мес"].isna()]["Вес 0 мес"] / self.df[self.df["ИМТ 0 мес"].isna()]["Рост"]**2).round()

        negative_target = ~pd.isna(self.df['% потери веса 3 мес']) \
                    & ~pd.isna(self.df['Вес 3 мес']) \
                    & (self.df['% потери веса 3 мес'] > 0) \
                    & ((self.df['Вес 0 мес']-self.df['Вес 3 мес'])<=0)
        self.df.loc[negative_target, '% потери веса 3 мес'] = 0# когда вес увеличился почему то в таблице положительные значения, здесь меняю на противополжое

        # self.df = self.df.loc[~pd.isna(self.df['Вес 3 мес']) | ~pd.isna(self.df['Вес 6 мес'])]
Dataset()
