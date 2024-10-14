"""this code is to analyze the already found results like below
thres;trigger_rate;synctatic_match;F1;partial_recall;partial_precision;synctatic_match_100tr;partial_recall_100tr;partial_precision_100tr;avg_pred_length;tes;avg_matched_prefix
0.000;0.00000;0.00000;0.00000;0.00000;0.00000;0.00000;0.0;0.00000;0.00000;0.00000;0.00000
0.100;0.00278;0.35714;0.00552;0.51020;0.72434;0.00099;0.0014196;0.00202;38.38040;0.00570;19.98671
0.199;0.00746;0.27588;0.01452;0.42278;0.67429;0.00206;0.0031519;0.00503;22.84439;0.01477;11.53751
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
from termcolor import colored


DATA = ["ddc", "dstc7"]
CONTEXTLENGTH = [None, "c2", "c4", "cinf"]
MODEL = ["t5", "gpt2"]
TYPE = ["seen", "unseen", "all"]

METRICS = ["synctatic_match", "partial_recall", "partial_precision"]
TRUNCATION_METRICS = METRICS + ['tes']
TRUNCATION = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

GRAPH_FOLDER = "meet"


def get_resfile(type, data, model, context, contextlength):
    if context:
        data = f"c{data}"
    if not contextlength:
        if model == "gpt2":
            model = "gpt2.sloss"
        return f"results/result.{type}.{data}.{model}.csv"
    return f"results/result.{type}.{data}.{model}.{contextlength}.csv"


def get_trunc_resfile(res_file, truncate):
    res_file = res_file.replace(".csv", f".truncate_{truncate}.csv")
    return res_file


def varying_context():
    for data in DATA:
        for model in MODEL:
            for type in TYPE:
                cols = ["contextlength"] + METRICS
                vals = []
                for contextlength in CONTEXTLENGTH:
                    resfile = get_resfile(
                        type, data, model, True, contextlength
                    )
                    # print(resfile)
                    df = pd.read_csv(resfile, sep=";")
                    # print(df.columns)
                    # get row corresponding to highest trigger_Rate
                    row = df.iloc[df["trigger_rate"].idxmax()]
                    vals.append(
                        [contextlength] + [row[metric] for metric in METRICS]
                    )
                    # print(df)
                df = pd.DataFrame(vals, columns=cols)
                graph_folder = GRAPH_FOLDER + f"/{type}_{data}_{model}"
                if os.path.exists(graph_folder) is False:
                    os.makedirs(graph_folder)
                df.to_csv(
                    f"{graph_folder}/varying_context.{type}.{data}.{model}.csv",
                    index=False,
                )
                print(
                    f"saved {graph_folder}/varying_context.{type}.{data}.{model}.csv"
                )
                # make graph
                for metric in METRICS:
                    plt.plot(
                        df["contextlength"].apply(lambda x: str(x)),
                        df[metric],
                        label=metric,
                    )
                plt.legend()
                plt.xlabel("context length")
                plt.ylabel("metric value")
                plt.title(f"{type} {data} {model}")
                plt.savefig(
                    f"{graph_folder}/varying_context.{type}.{data}.{model}.png"
                )
                plt.close()
                print(
                    f"saved {graph_folder}/varying_context.{type}.{data}.{model}.png"
                )


def varying_truncation():
    files = list(
        filter(
            lambda x: "prefix" not in x and x.endswith(".csv"),
            os.listdir("results"),
        )
    )
    # print(files)
    res_files = list(
        filter(lambda x: x.startswith("result") and "truncate" not in x, files)
    )
    trunc_files = list(
        filter(lambda x: x.startswith("result") and "truncate" in x, files)
    )
    # print("result.unseen.cddc.gpt4.truncate_1.csv" in trunc_files)
    done = []
    for res_file in res_files:
        cols = ["truncate"] + TRUNCATION_METRICS
        vals = []
        for truncate in TRUNCATION:
            truncate_res_file = get_trunc_resfile(res_file, truncate)
            if truncate_res_file not in trunc_files:
                continue
            tdf = pd.read_csv("results/"+truncate_res_file, sep=";")
            # get row corresponding to highest trigger_Rate
            row = tdf.iloc[tdf["trigger_rate"].idxmax()]
            vals.append([truncate] + [row[metric] for metric in TRUNCATION_METRICS])
        if vals == []:
            continue
        df = pd.DataFrame(vals, columns=cols)
        graph_folder = GRAPH_FOLDER + f"/truncation"
        if os.path.exists(graph_folder) is False:
            os.makedirs(graph_folder)
        df.to_csv(
            f"{graph_folder}/{res_file.replace('.csv', '')}.truncate_all.csv",
            index=False,
        )
        print(
            colored(
                f"saved {graph_folder}/{res_file.replace('.csv', '')}.truncate_all.csv",
                "green",
            )
        )
        done.append(res_file)
    print("done files: ", done)


if __name__ == "__main__":
    varying_context()
    # varying_truncation()
