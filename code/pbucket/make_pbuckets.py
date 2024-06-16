import numpy as np
import argparse
import os
from tqdm import tqdm

_prefix_bucket_ranges = [(1, 5), (6, 20), (20, np.inf)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str, default="../anal/outputs")
    parser.add_argument("--output_folder", type=str, default="Poutputs")
    args = parser.parse_args()
    files = [i for i in os.listdir(args.input_folder) if "all" in i]
    for i in tqdm(files):
        p_buckets = [[] for i in _prefix_bucket_ranges]
        with open(os.path.join(args.input_folder, i), "r") as f:
            for line in f:
                line = line.strip().split("\t")
                if(len(line)!=5):
                    continue
                prefix = line[0]
                act_prefix = prefix.lower()
                act_prefix = act_prefix.split("<|eou|>")[-1]
                act_prefix = act_prefix.split("<eou>")[-1]            
                for idx, (start, end) in enumerate(_prefix_bucket_ranges):
                    if(start<=len(act_prefix)<=end):
                        p_buckets[idx].append(line)
                        break
        for idx, bucket in enumerate(p_buckets):
            with open(os.path.join(args.output_folder, (i.replace(".all", ".PB{}".format(idx)))), "w") as f:
                for line in bucket:
                    f.write("\t".join(line) + "\n")