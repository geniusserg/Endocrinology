import pandas as pd
import model
from train_model import Dataset
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind, pearsonr, spearmanr
import numpy as np
import os

def get_conditions(dataset, threshold=5, condition="A"):
  conditions = {
      "A": (dataset["% потери веса 3 мес"]>=threshold),
      "B": (dataset[~pd.isna(dataset["% потери веса 6 мес"])]["% потери веса 6 мес"]>=threshold),
      'C': (np.nanmax(dataset[["% потери веса 3 мес","% потери веса 6 мес"]], axis=1)>=threshold),
      'D': (dataset["% потери веса 3 мес"]>=threshold)&(dataset["% потери веса 6 мес"]>=threshold)
  }
  return conditions[condition]

def test_chisquare(dt, threshold=5, condition="A"):
    dataset_x = dt.dataset
    dataset_x["success"] = get_conditions(dataset_x, threshold, condition)
    stats = []
    for col in dataset_x.columns:
        res = [col, None, None, None]
        if (col == "success"):
            continue
        if ((~(dataset_x[col].isna())).sum() < 20):
            continue
        try:
            A = dataset_x.pivot_table(columns=col, index="success", aggfunc="size").fillna(0)
        except:
            raise Exception(f"{col} cannot be calculated")
        try:
            res[1] = chi2_contingency(A)[1]
        except Exception as e:
            pass
        try:
            res[2] = fisher_exact(A)[1]
        except Exception as e:
            pass
        res[3] = ((~(dataset_x[col].isna())).sum())
        stats.append(res)
    dfds = pd.DataFrame(stats, columns=["Param", "p-value chisquare", "p-value fisher", "number"])
    dfds = dfds[((~(dfds["p-value chisquare"].isna()))&(dfds["p-value chisquare"] <= 0.05))
                |((~(dfds["p-value fisher"].isna()))&(dfds["p-value fisher"]<=0.05))].reset_index(drop=True)
    dfds.to_csv(f"discret_test_{threshold}_{condition}.csv")
    return dfds

def test_distributions(dt, threshold=5, condition="A"):
    dataset_x = dt.dataset
    dataset_x["success"] = get_conditions(dataset_x, threshold, condition)

    dcolsnunique = dataset_x.nunique()
    dcolsnunique = dcolsnunique[dcolsnunique>3].index.values

    stats = []

    for p in dcolsnunique:
        if (p == "success"):
            continue
        param_not_none = dataset_x[~dataset_x[p].isna()]
        if (param_not_none.shape[0] < 10):
            continue
        if any([p.find(prf)!=-1 for prf in [ "6 мес", "В2", "Респондер", "В.2", "В1..1", "ПЭТ"]]):
            continue
        param_from_fail = param_not_none[param_not_none["success"]==0][p]
        param_from_succ = param_not_none[param_not_none["success"]==1][p]
        try:
            mw_result = mannwhitneyu(param_from_fail, param_from_succ)
        except:
            mw_result = [1, 1]
        try:
            ttest_result = ttest_ind(param_from_fail, param_from_succ)
        except:
            ttest_result = [1, 1]
        stats.append([p, mw_result[1], ttest_result[1], param_from_succ.mean(), param_from_succ.std(), param_from_fail.mean(), param_from_fail.std(ddof=0), param_not_none.shape[0]])
    ks_stats = pd.DataFrame(stats, columns=["Параметр", "MW p-value", "t-test p_value", '+ mean', '+ std', '- mean', '- std', "Количество сэмплов"]).round(2)
    ks_stats = ks_stats[((~(ks_stats["MW p-value"].isna()))&(ks_stats["MW p-value"]<=0.05))|((~(ks_stats["t-test p_value"].isna()))&(ks_stats["t-test p_value"]<=0.05))].reset_index(drop=True)
    ks_stats.to_csv(f"distribution_test_{threshold}_{condition}.csv")
    return ks_stats


def test_correlation(dt, condition="A"):
    dataset_x = dt.dataset
    x = dataset_x
    if (condition=="A"):
        y = x["% потери веса 3 мес"].to_numpy()
        x = x.drop("% потери веса 3 мес", axis=1)
    if (condition=="B"):
        x = x[~x["% потери веса 6 мес"].isna()]
        y = x["% потери веса 6 мес"].to_numpy()
        x = x.drop("% потери веса 6 мес", axis=1)
    dcolsnunique = dataset_x.nunique()
    dcolsnunique = dcolsnunique[dcolsnunique<5].index.values
    x = x.loc[:, ~x.isna().all()]
    stats = []
    for p in x.columns:
        mask = ~(x[p].isna())
        param_not_none = x[mask]
        if (param_not_none.shape[0] < 10):
            continue
        if any([p.find(prf)!=-1 for prf in [ "6 мес", "В2", "Респондер", "В.2", "В1..1", "ПЭТ"]]) or (p in dcolsnunique):
            continue
        res = [p, None, None, None, None]
        res[1:3] = pearsonr(np.array(x[mask][p]), y[mask])
        res[3:] = spearmanr(np.array(x[mask][p]), y[mask])
        stats.append(res)
    corr_stats = pd.DataFrame(stats, columns=["Параметр", "pearson", "pearson p-value", "spearman", "spearman p-value"]).round(2)
    corr_stats = corr_stats[((~(corr_stats["pearson p-value"].isna()))&(corr_stats["pearson p-value"]<=0.05))
                            |((~(corr_stats["spearman p-value"].isna()))&(corr_stats["spearman p-value"]<=0.05))].reset_index(drop=True)
    corr_stats.to_csv(f"correlation_test_{condition}.csv")
    return corr_stats
    


dt = Dataset(dataset_path=os.path.join("..", "data", "dataset.xlsx"))
dt.preprocess(medicine="SIB", agroup_params_only=False)


test_chisquare(dt, 5, "A")
test_chisquare(dt, 7, "B")
test_distributions(dt, 5, "A")
test_distributions(dt, 7, "B")
test_correlation(dt, "A")
test_correlation(dt, "B")