import pandas as pd
import numpy as np

models = [
    "qb",
    "t5",
    "gpt2.sloss",
    "mistral.prompt.word2",
    "phi.prompt.word10",
    "phi.finetune.word3",
    "gpt4",
]

metrics = [
    "trigger_rate",
    "synctatic_match",
    "partial_recall",
    "partial_precision",
    "avg_pred_length",
    "thres",
    "tes",
]
datas = ["ddc", "dstc7"]


def predict_result_file(model, data, context):
    res = "results/result.all."
    if context:
        res += "c"
    res += data
    res += "."
    res += model
    res += ".csv"
    return res


def rename(x):
    x = x.replace(".sloss", "")
    x = x.replace(".prompt.word2", "prompt")
    x = x.replace(".prompt.word10", "prompt")
    x = x.replace(".finetune.word3", "finetune")
    return x


def make_table(metric, data):
    table = []
    columns = []
    for model in models:
        try:
            x = predict_result_file(model, data, False)
            df = pd.read_csv(x, sep=";")
            df["trigger_rate"] = df["trigger_rate"].apply(lambda x: float(x))
            df = df.sort_values(by=["trigger_rate"])
            print(len(df))
            if len(df) == 301:
                columns.append(rename(model))
                table.append(np.array(df[metric]))
        except Exception as e:
            print(e)
            print("occured at " + model, metric, data)
    for model in models:
        try:
            x = predict_result_file(model, data, True)
            df = pd.read_csv(x, sep=";")
            print(len(df))
            if len(df) == 301:
                columns.append("context + " + rename(model))
                table.append(np.array(df[metric]))
        except Exception as e:
            print(e)
            print("occured at " + model, metric, data)
    return pd.DataFrame(np.array(table).T, columns=columns)


if __name__ == "__main__":
    for metric in metrics:
        for data in datas:
            df = make_table(metric, data)
            df.to_csv("tables/" + metric + "." + data + ".csv", sep=",", index=False)
            print("saved ", metric, data)
