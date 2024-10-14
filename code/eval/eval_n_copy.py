import sys

sys.path.append("code/eval")

import argparse
import tqdm
from tes import TES
import numpy as np
import thres_ckecks
import pandas as pd
from string import punctuation
import context_correction

"""
take queries (x), gt (y), predictions (pred), costs (cost) and subword_len(if not like QB, then put 1) from stdin
then filter output based on threshold

"""
min_max = {
    "QB": (0, 30),
    "GPT2": (0, 30),
    "T5": (0, 30),
    "PHI_PROMPT": (0, 1),
    "PHI_FINETUNE": (0, 5),
    "MISTRAL": (0, 1),
    "NGAME": (0, 5),
    "GPT4": (0, 5),
    "rBERT": (0, 5),
}

_space_correction_models = [
    "GPT2",
    "T5",
]
_trailing_punctuation_correction_models = [
    "QB",
    "RENEE",
]
_gt_empty_correction_models = ["GPT2", "MISTRAL"]
# _num_cost_models = ['QB', 'GPT2', 'T5', 'PHI_PROMPT', 'PHI_FINETUNE', 'MISTRAL', 'NGAME', 'rBERT']


def min_max_to_thresholds(min, max):
    # 300 thresholds between min max
    h = (max - min) / 301
    return [min + i * h for i in range(301)]


mapping = {
    "QB": (thres_ckecks.thres_checkQB, min_max_to_thresholds(*min_max["QB"])),
    "MPC": (thres_ckecks.thres_checkMPC, [0]),
    "GPT2": (
        thres_ckecks.thres_checkGPT2,
        min_max_to_thresholds(*min_max["GPT2"]),
    ),
    "T5": (
        thres_ckecks.thres_checkGPT2,
        min_max_to_thresholds(*min_max["T5"]),
    ),
    "PHI_PROMPT": (
        thres_ckecks.thres_checkGPT2,
        min_max_to_thresholds(*min_max["PHI_PROMPT"]),
    ),
    "PHI_FINETUNE": (
        thres_ckecks.thres_checkGPT2,
        min_max_to_thresholds(*min_max["PHI_FINETUNE"]),
    ),
    "MISTRAL": (
        thres_ckecks.thres_checkGPT2,
        min_max_to_thresholds(*min_max["MISTRAL"]),
    ),
    "NGAME": (
        thres_ckecks.thres_checkNGAME,
        min_max_to_thresholds(*min_max["NGAME"]),
    ),
    "GPT4": (thres_ckecks.thres_checkGPT4, [0]),
    "RENEE": (thres_ckecks.thres_checkMPC, [0]),
    # 0.05 intervel from 0 to 0.9, 0.0005 intervel from 0.9 to 1
    # 'rBERT': (thres_ckecks.thres_checkNGAME, [i/1000 for i in range(100)]+[8*i/1000 + 0.1 for i in range(100)] + [0.9 + i/1000 for i in range(101)]),
}


# mapping['GPT2'] = (thres_ckecks.thres_checkGPT2, [i*0.1 for i in range(299, 300)])


def space_correction(x):
    x = x.replace(" ,", ",")
    x = x.replace(" .", ".")
    x = x.replace(" ?", "?")
    x = x.replace(" !", "!")
    x = x.replace(" ’", "’")
    return x


def longest_Common_Prefix(str1):

    if not str1:  # Check if the list is empty
        return ""

    short_str = min(str1, key=len)  # Find the shortest string in the list

    for i, char in enumerate(
        short_str
    ):  # Iterate through characters in the shortest string
        for other in str1:  # Iterate through other strings in the list
            if (
                other[i] != char
            ):  # Check if the corresponding characters don't match
                return short_str[:i]  # Return the common prefix found so far

    return (
        short_str  # Return the entire shortest string if it is a common prefix
    )


class Bucket:
    def __init__(self, ls=[], total_lines=0, model=""):
        # print(ls[:10])
        self.ls = ls
        self.total_lines = total_lines
        self.missed = total_lines - len(ls)
        self.exact = 0
        self.total_recall = 0
        self.total_precision = 0
        self.total_prediction_length = 0
        self.total_common_prefix = 0
        self.model = model
        for i in self.ls:
            if i[1] == i[2]:
                self.exact += 1
            common_prefix = longest_Common_Prefix([i[1], i[2]])
            self.total_common_prefix += len(common_prefix)
            self.total_recall += len(common_prefix) / len(i[1])
            self.total_precision += len(common_prefix) / len(i[2])
            self.total_prediction_length += len(i[2])

    def merge(self, other):
        self.ls.extend(other.ls)
        self.missed = self.total_lines - len(self.ls)
        self.exact += other.exact
        self.total_common_prefix += other.total_common_prefix
        self.total_recall += other.total_recall
        self.total_precision += other.total_precision
        self.total_prediction_length += other.total_prediction_length

    def compute(self):
        # assuming already merged
        try:
            return {
                "rc_maxtr": self.total_recall
                / (self.total_lines - self.missed),
                "pr_maxtr": self.total_precision
                / (self.total_lines - self.missed),
                "al": self.total_prediction_length
                / (self.total_lines - self.missed),
                "rc_100tr": self.total_recall / self.total_lines,
                "pr_100tr": self.total_precision / self.total_lines,
                "sm": self.exact / (self.total_lines - self.missed),
                "sm_100tr": self.exact / self.total_lines,
                "tr": len(self.ls) / self.total_lines,
                "avg_matched_prefix": self.total_common_prefix
                / (self.total_lines - self.missed),
                "total_lines": self.total_lines,
            }
        except ZeroDivisionError:
            return {
                "rc_maxtr": 0.0,
                "pr_maxtr": 0.0,
                "al": 0.0,
                "rc_100tr": 0.0,
                "pr_100tr": 0.0,
                "sm": 0.0,
                "sm_100tr": 0.0,
                "tr": 0.0,
                "avg_matched_prefix": 0.0,
                "total_lines": self.total_lines,
            }

        # tr = 1 - missed[i] / tot_lines
        # sm = exact[i] / (tot_lines - missed[i])
        # rc = precision[i] / (tot_lines - missed[i])
        # pr = recall[i] / (tot_lines - missed[i])
        # sm_100tr = exact[i] / tot_lines
        # rc_100tr = precision[i] / tot_lines
        # pr_100tr = recall[i] / tot_lines
        # al = pred_lengths[i] / (tot_lines - missed[i])


def req_space_correction(model):
    return model in _space_correction_models


def preprocess(i: str, args):
    i = i.lower()
    i = i.strip("\n")
    # print(i)
    x, y, pred, cost, subword_len = i.split("\t")
    pred = pred.replace("<|eou|>", "")
    x = x.split("<|eou|>")[-1]
    x = x.split("<eou>")[-1]
    y.replace("<|eou|>", "")
    y.replace("<eou>", "")
    x = x.lstrip()
    y = y.rstrip()
    pred = pred.rstrip()
    # NLG tokenizer temporary fix
    if req_space_correction(args.model):
        # remove single space before punctuation
        y = space_correction(y)
        pred = space_correction(pred)
        x = space_correction(x)
    if args.model in _trailing_punctuation_correction_models and has_context(
        args.input
    ):
        # remove last punctuation for models that donot predict it
        y = y.rstrip(punctuation)
        pred = pred.rstrip(punctuation)
        y = y.rstrip()
        pred = pred.rstrip()

    return [x, y, pred, cost, subword_len]


def has_context(x):
    x = x.lower()
    return "cddc" in x or "cdstc7" in x


def truncate(x: str, n: int) -> str:
    x = x.split(" ")
    x = x[:n]
    return " ".join(x)


def get_max_context(x):
    if x.endswith("c2"):
        return 2
    if x.endswith("c4"):
        return 4
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--truncate", type=int, default=None)
    parser.add_argument("--only_max", action="store_true")
    args = parser.parse_args()
    thres_check, thresholds = mapping[args.model]
    if args.only_max:
        thresholds = [thresholds[-1]]
    max_context = get_max_context(args.input)
    tes = TES(args.input, args.truncate, max_context)
    ls = []
    with open(args.input) as f:
        ls = f.readlines()
    ls = list(filter(lambda x: len(x.split("\t")) == 5, ls))
    if has_context(args.input):
        ls, _ = context_correction.correct(ls)
    ls = [preprocess(i, args) for i in ls]
    # if(args.model == 'QB' and has_context(args.input)):
    if (
        args.model in _trailing_punctuation_correction_models
        or args.model in _gt_empty_correction_models
    ):
        # might have created empty ground truth. just remove them
        ls = list(filter(lambda x: not len(x[1]) == 0, ls))
    if args.truncate:
        ls = [
            [i[0], i[1], truncate(i[2], args.truncate), i[3], i[4]] for i in ls
        ]
    # print(ls)
    # exit(0)
    df = pd.DataFrame(
        ls, columns=["prefix", "gt", "pred", "cost", "subword_len"]
    )
    # print(len(ls), len(df))
    threshold_buckets = [[] for i in thresholds]
    # print("making buckets")
    for i in tqdm.tqdm(df.values):
        # assume threscheck is monotone function.
        # get the first threshold where threscheck is true
        lo = 0
        hi = len(thresholds) - 1
        res = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if not thres_check(*i, thresholds[mid]):
                lo = mid + 1
            else:
                hi = mid - 1
                res = mid
        if res != -1:
            threshold_buckets[res].append(i)
    # print("threshold_buckets", [len(i) for i in threshold_buckets])
    buckets = [Bucket(i, len(df), args.model) for i in threshold_buckets]
    # exit(0)
    prefix_buckets = [buckets[0]]
    results = [buckets[0].compute()]
    print("merging")
    for i in tqdm.tqdm(range(1, len(buckets))):
        prefix_buckets.append(prefix_buckets[-1])
        prefix_buckets[-1].merge(buckets[i])
        results.append(prefix_buckets[-1].compute())
    print("calculating tes")
    for idx, thres in enumerate(tqdm.tqdm(thresholds)):
        results[idx]["tes"] = np.nanmean(tes.run(thres)[:, 0])
    # results = [i.compute() for i in prefix_buckets]
    # print(results[-1])
    with open(args.output, "w+") as f:
        f.write(
            "thres;trigger_rate;synctatic_match;F1;partial_recall;partial_precision;synctatic_match_100tr;partial_recall_100tr;partial_precision_100tr;avg_pred_length;tes;avg_matched_prefix\n"
        )
    for i, threshold in enumerate(thresholds):
        with open(args.output, "a") as f:
            tr = results[i]["tr"]
            sm = results[i]["sm"]
            rc = results[i]["rc_maxtr"]
            pr = results[i]["pr_maxtr"]
            sm_100tr = results[i]["sm_100tr"]
            rc_100tr = results[i]["rc_100tr"]
            pr_100tr = results[i]["pr_100tr"]
            al = results[i]["al"]
            tes = results[i]["tes"]
            avg_matched_prefix = results[i]["avg_matched_prefix"]
            f1 = 2 * sm * tr / (sm + tr) if sm + tr != 0 else 0.0
            f.write(
                "{:.3f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5};{:.5f};{:.5f};{:.5f};{:.5f}\n".format(
                    threshold,
                    tr,
                    sm,
                    f1,
                    rc,
                    pr,
                    sm_100tr,
                    rc_100tr,
                    pr_100tr,
                    al,
                    tes,
                    avg_matched_prefix,
                )
            )


if __name__ == "__main__":
    main()