import sys
import argparse
import random
import io
import os
import time
import tqdm
import tes_ as tes_lib
import numpy as np
import thres_ckecks
import pandas as pd
from multiprocessing import Pool
from functools import partial
from copy import deepcopy
"""
take queries (x), gt (y), predictions (pred), costs (cost) and subword_len(if not like QB, then put 1) from stdin
then filter output based on threshold

"""
min_max = {
    'QB': (0, 30),
    'GPT2': (0, 30),
    'T5': (0, 30),
    'PHI_PROMPT': (0, 1),
    'PHI_FINETUNE': (0, 5),
    'MISTRAL': (0, 1),
    'NGAME': (0, 5),
    'GPT4': (0, 5),
    'rBERT': (0, 5),
}

_space_correction_models = ['GPT2', 'T5', 'PHI_PROMPT', 'PHI_FINETUNE', 'MISTRAL', 'rBERT']
# _num_cost_models = ['QB', 'GPT2', 'T5', 'PHI_PROMPT', 'PHI_FINETUNE', 'MISTRAL', 'NGAME', 'rBERT']

def min_max_to_thresholds(min, max):
    # 300 thresholds between min max
    h = (max - min) / 301
    return [min + i*h for i in range(301)]

mapping = {
    'QB': (thres_ckecks.thres_checkQB, min_max_to_thresholds(*min_max['QB'])),
    'MPC': (thres_ckecks.thres_checkMPC, [0]),
    'GPT2': (thres_ckecks.thres_checkGPT2, min_max_to_thresholds(*min_max['GPT2'])),
    'T5': (thres_ckecks.thres_checkGPT2, min_max_to_thresholds(*min_max['T5'])),
    'PHI_PROMPT': (thres_ckecks.thres_checkGPT2, min_max_to_thresholds(*min_max['PHI_PROMPT'])),
    'PHI_FINETUNE': (thres_ckecks.thres_checkGPT2, min_max_to_thresholds(*min_max['PHI_FINETUNE'])),
    'MISTRAL': (thres_ckecks.thres_checkGPT2, min_max_to_thresholds(*min_max['MISTRAL'])),
    'NGAME': (thres_ckecks.thres_checkNGAME, min_max_to_thresholds(*min_max['NGAME'])),
    'GPT4': (thres_ckecks.thres_checkGPT4, [0]),
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

    for i, char in enumerate(short_str):  # Iterate through characters in the shortest string
        for other in str1:  # Iterate through other strings in the list
            if other[i] != char:  # Check if the corresponding characters don't match
                return short_str[:i]  # Return the common prefix found so far

    return short_str  # Return the entire shortest string if it is a common prefix

from string import punctuation


def dfr_gen(i, df, thres_check):
    print("hello from {}".format(i))
    dfr = df[df.apply(lambda x:thres_check(*x, i), axis = 1)]
    print("bye from {}".format(i))


def longest_Common_Prefix(str1):
    
    if not str1:  # Check if the list is empty
        return ""

    short_str = min(str1, key=len)  # Find the shortest string in the list

    for i, char in enumerate(short_str):  # Iterate through characters in the shortest string
        for other in str1:  # Iterate through other strings in the list
            if other[i] != char:  # Check if the corresponding characters don't match
                return short_str[:i]  # Return the common prefix found so far

    return short_str  # Return the entire shortest string if it is a common prefix

class Bucket:
    def __init__(self, ls, total_lines, raw_test_file, model):
        # print(ls[:10])
        self.ls = ls
        self.total_lines = total_lines
        self.missed = total_lines - len(ls)
        self.exact = 0
        self.total_recall = 0
        self.total_precision = 0
        self.total_prediction_length = 0
        # self.tes = 0
        self.raw_test_file = raw_test_file
        self.model = model
        for i in self.ls:
            if(i[1]==i[2]):
                self.exact+=1
            common_prefix = longest_Common_Prefix([i[1], i[2]])
            self.total_recall+=len(common_prefix) / len(i[1])
            self.total_precision+=len(common_prefix) / len(i[2])
            self.total_prediction_length+=len(i[2])

    def merge(self, other):
        self.ls.extend(other.ls)
        self.missed = self.total_lines - len(self.ls)
        self.exact += other.exact
        self.total_recall += other.total_recall
        self.total_precision += other.total_precision
        self.total_prediction_length += other.total_prediction_length
    

    def compute(self):
        # assuming already merged
        if(self.ls != []):
            dfr = pd.DataFrame(np.array(self.ls, ndmin=2)[:, [0, 2]], columns = ['prefix', 'prediction'])
            # tes = tes_lib.tes_compute(dfr, self.raw_test_file, self.model)
        try:
            return {
                'rc_maxtr': self.total_recall / (self.total_lines - self.missed),
                'pr_maxtr': self.total_precision / (self.total_lines - self.missed),
                'al': self.total_prediction_length / (self.total_lines - self.missed),
                'rc_100tr': self.total_recall / self.total_lines,
                'pr_100tr': self.total_precision / self.total_lines,
                'sm': self.exact / (self.total_lines - self.missed),
                'sm_100tr': self.exact / self.total_lines,
                'tr': len(self.ls) / self.total_lines
            }
        except ZeroDivisionError:
            return {
                'rc_maxtr': 0.0,
                'pr_maxtr': 0.0,
                'al': 0.0,
                'rc_100tr': 0.0,
                'pr_100tr': 0.0,
                'sm': 0.0,
                'sm_100tr': 0.0,
                'tr': 0.0
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


def preprocess(i, args):
    i = i.lower()
    i = i.strip("\n")
    # print(i)
    x, y, pred, cost, subword_len = i.split('\t')
    pred = pred.replace("<|eou|>", "")
    x = x.split("<|eou|>")[-1]
    x = x.split("<eou>")[-1]
    x = x + "x"
    y = "x" + y
    pred = "x" + pred
    y.replace("<|eou|>", "")
    y.replace("<eou>", "")
    x = x.strip()
    y = y.strip()
    pred = pred.strip()
    x = x[:-1]
    y = y[1:]
    pred = pred[1:]
    # NLG tokenizer temporary fix
    if(req_space_correction(args.model)):
        #remove single space before punctuation
        y = space_correction(y)
        pred = space_correction(pred)
        x = space_correction(x)
    if(args.model == 'QB'):
        y = "x" + y
        pred = "x" + pred
        y = y.strip(punctuation)
        pred = pred.strip(punctuation)
        y = y.strip()
        pred = pred.strip()
        y = y[1:]
        pred = pred[1:]

    return [x, y, pred, cost, subword_len]


def predict_raw_test_File(input_file):
    # print(input_file)
    input_file = input_file.lower()
    input_file = input_file.strip("\n")
    if('unseen' in input_file):
        if("cddc" in input_file):
            return "data/cDDC/unseen/test_formatted.txt"
        if("cdstc7" in input_file):
            return "data/cDSTC7/unseen/test_formatted.txt"
        if("ddc" in input_file):
            return "data/DDC/unseen/test_formatted.txt"
        if("dstc7" in input_file):
            return "data/DSTC7/unseen/test_formatted.txt"
        raise ValueError("Invalid input file")
    if('seen' in input_file):
        if("cddc" in input_file):
            return "data/cDDC/seen/test_formatted.txt"
        if("cdstc7" in input_file):
            return "data/cDSTC7/seen/test_formatted.txt"
        if("ddc" in input_file):
            return "data/DDC/seen/test_formatted.txt"
        if("dstc7" in input_file):
            return "data/DSTC7/seen/test_formatted.txt"
    if('all' in input_file):
        if("cddc" in input_file):
            return "data/cDDC/all/test_formatted.txt"
        if("cdstc7" in input_file):
            return "data/cDSTC7/all/test_formatted.txt"
        if("ddc" in input_file):
            return "data/DDC/all/test_formatted.txt"
        if("dstc7" in input_file):
            return "data/DSTC7/all/test_formatted.txt"
    raise ValueError("Invalid input file")

def has_context(x):
    x = x.lower()
    return "cddc" in x or "cdstc7" in x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    thres_check, thresholds = mapping[args.model]
    raw_test_file = None
    # predict_raw_test_File(args.input)
    ls = []
    with open(args.input) as f:
        ls = f.readlines()
    ls = list(filter(lambda x:len(x.split("\t"))==5, ls))
    ls = [preprocess(i, args) for i in ls]
    # if(args.model == 'QB' and has_context(args.input)):
    ls = list(filter(lambda x:not (len(x[1])==0), ls))
    # print(ls)
    # exit(0)
    df = pd.DataFrame(ls, columns = ['prefix', 'gt', 'pred', 'cost', 'subword_len'])
    threshold_buckets = [[] for i in thresholds]
    # print("making buckets")
    for i in tqdm.tqdm(df.values):
        # assume threscheck is monotone function.
        # get the first threshold where threscheck is true
        lo = 0
        hi = len(thresholds)-1
        res = -1
        while(lo<=hi):
            mid = (lo+hi)//2
            if(not thres_check(*i, thresholds[mid])):
                lo = mid+1
            else:
                hi = mid-1
                res = mid
        if(res!=-1):
            threshold_buckets[res].append(i)
    # print("threshold_buckets", [len(i) for i in threshold_buckets])
    buckets = [Bucket(i, len(df), raw_test_file, args.model) for i in threshold_buckets]
    # exit(0)
    prefix_buckets = [buckets[0]]
    results = [buckets[0].compute()]
    print("merging")
    for i in tqdm.tqdm(range(1, len(buckets))):
        prefix_buckets.append(prefix_buckets[-1])
        prefix_buckets[-1].merge(buckets[i])
        results.append(prefix_buckets[-1].compute())
    # results = [i.compute() for i in prefix_buckets]
    # print(results[-1])
    with open(args.output, 'w') as f:
        f.write("thres;trigger_rate;synctatic_match;F1;partial_recall;partial_precision;synctatic_match_100tr;partial_recall_100tr;partial_precision_100tr;avg_pred_length\n")
    for i, threshold in enumerate(thresholds):
        with open(args.output, "a") as f:
            tr = results[i]['tr']
            sm = results[i]['sm']
            rc = results[i]['rc_maxtr']
            pr = results[i]['pr_maxtr']
            sm_100tr = results[i]['sm_100tr']
            rc_100tr = results[i]['rc_100tr']
            pr_100tr = results[i]['pr_100tr']
            al = results[i]['al']
            f1 = 2*sm*tr/(sm+tr) if sm+tr!=0 else 0.0
            f.write("{:.3f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5};{:.5f};{:.5f}\n".format(threshold, tr, sm, f1, rc, pr, sm_100tr, rc_100tr, pr_100tr, al))
if __name__ == '__main__':
    main()
