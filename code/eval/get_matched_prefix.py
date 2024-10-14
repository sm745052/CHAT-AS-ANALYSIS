import argparse
import pandas as pd
import tqdm
from eval_n_copy import Bucket, preprocess, predict_raw_test_File, has_context, mapping
import math
import os
import context_correction

def get_shortened_pred(threshold, output_file, input_file):
    with open(input_file, 'r') as infile:
        with open(output_file, 'w+') as outfile:
            for line in infile:
                if len(line.split('\t')) == 5:
                    prefix, gt, pred, score, subword = line.split('\t')
                    new_pred = pred[:threshold]
                    outfile.write(f'{prefix}\t{gt}\t{new_pred}\t{score}\t{subword}')
    print("written into outfile...")
    
    
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, required=True)
parser.add_argument('--output_folder', type=str, required=True)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

thres_check, thresholds = mapping[args.model]

ls = []
with open(args.input_file) as f:
    ls = f.readlines()
ls = list(filter(lambda x: len(x.split("\t")) == 5, ls))
if has_context(args.input_file):
    ls, _ = context_correction.correct(ls)
ls = [preprocess(i, args) for i in ls]

df = pd.DataFrame(ls, columns=["prefix", "gt", "pred", "cost", "subword_len"])

threshold_buckets = [[] for _ in thresholds]

for i in tqdm.tqdm(df.values):
    lo, hi = 0, len(thresholds) - 1
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

raw_test_file = predict_raw_test_File(args.input_file)
buckets = [Bucket(i, len(df), raw_test_file, args.model) for i in threshold_buckets]

prefix_buckets = [buckets[0]]
results = [buckets[0].compute()]

for i in tqdm.tqdm(range(1, len(buckets))):
    prefix_buckets.append(prefix_buckets[-1])
    prefix_buckets[-1].merge(buckets[i])
    results.append(prefix_buckets[-1].compute())
    
threshold = math.ceil(results[-1]["avg_matched_prefix"])
print(f'Threshold: {threshold}, Matched Length: {results[-1]["avg_matched_prefix"]}')
outfile = os.path.join(args.output_folder, os.path.basename(args.input_file)+f'.L{threshold}')
get_shortened_pred(threshold, outfile, args.input_file)


