import sys
import argparse
import random
import io
import os
import tqdm
import thres_ckecks
"""
take queries (x), gt (y), predictions (pred), costs (cost) and subword_len(if not like QB, then put 1) from stdin
then filter output based on threshold

"""



mapping = {
    'QB': (thres_ckecks.thres_checkQB, [i*0.1 for i in range(1, 300)]),
    'MPC': (thres_ckecks.thres_checkMPC, [0]),
    'GPT2': (thres_ckecks.thres_checkGPT2, [i*0.1 for i in range(1, 300)]),
    'NGAME': (thres_ckecks.thres_checkNGAME, [i*0.05 for i in range(0, 20)]),
    'GPT4': (thres_ckecks.thres_checkGPT4, [0]),
    # 0.05 intervel from 0 to 0.9, 0.0005 intervel from 0.9 to 1
    'rBERT': (thres_ckecks.thres_checkNGAME, [i/1000 for i in range(100)]+[8*i/1000 + 0.1 for i in range(100)] + [0.9 + i/1000 for i in range(101)]),
}

# mapping['GPT2'] = (thres_ckecks.thres_checkGPT2, [i*0.1 for i in range(299, 300)])


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

def main():
    tot_lines = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    args = parser.parse_args()
    thres_ckeck, thresholds = mapping[args.model]
    missed = [0] * len(thresholds)
    exact = [0] * len(thresholds)
    matched_length = [0] * len(thresholds)
    precision = [0] * len(thresholds)
    recall = [0] * len(thresholds)
    pred_lengths = [0] * len(thresholds)
    with open(args.output, 'w') as f:
        f.write("thres;trigger_rate;synctatic_match;F1;matched_length;saved_keystrokes;partial_recall;partial_precision;synctatic_match_100tr;partial_recall_100tr;partial_precision_100tr;avg_pred_length\n")
    with open(args.input, 'r') as f:
        for query in tqdm.tqdm(f):
            ##temp
            tot_lines += 1
            query = query.strip()
            if(query.split('\t')[-1] != '-'):
                try:
                    x, y, pred, cost, subword_len = query.split('\t')
                    # remove EOU temporary fix for GPT2 sloss without context
                    pred = pred.replace("<|EOU|>", "")
                    # NLG tokenizer temporary fix
                    if(args.model == "GPT2" or "rBERT" in args.model):
                        #remove single space before punctuation
                        y = y.replace(" ,", ",")
                        y = y.replace(" .", ".")
                        y = y.replace(" ?", "?")
                        y = y.replace(" !", "!")
                    
                except Exception as e:
                    print(e)
                    print(query.split("\t"))
            for i, threshold in enumerate(thresholds):
                if(query.split('\t')[-1] =='-'):
                    missed[i] += 1
                    continue
                if thres_ckeck(x, y, pred, cost, subword_len, threshold):
                    pred_lengths[i] += len(pred)
                    # check without case TO_DO
                    y = y.lower()
                    pred = pred.lower()
                    common = longest_Common_Prefix([y, pred])
                    try:
                        precision[i] += len(common)/len(y)
                        recall[i] += len(common)/len(pred)
                    except Exception as e:
                        print(e)
                        print(args.input)
                        print(len(y), len(pred))
                        print(thres_ckeck(x, y, pred, cost, subword_len, threshold))
                    if pred == y:
                        exact[i] += 1
                        matched_length[i] += len(y)
                else:
                    missed[i] += 1
    for i, threshold in enumerate(thresholds):
        with open(args.output, "a") as f:
            if exact[i] == 0:
                f.write("{};{};{};{};{};{};{};{};{};{};{};{}\n".format(threshold, 1 - missed[i] / tot_lines, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
            else:
                tr = 1 - missed[i] / tot_lines
                sm = exact[i] / (tot_lines - missed[i])
                rc = precision[i] / (tot_lines - missed[i])
                pr = recall[i] / (tot_lines - missed[i])
                sm_100tr = exact[i] / tot_lines
                rc_100tr = 0 / tot_lines
                pr_100tr = recall[i] / tot_lines
                al = pred_lengths[i] / (tot_lines - missed[i])
                f.write("{:.3f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5f};{:.5};{:.5f};{:.5f};{:.5};{:.5}\n".format(threshold, tr, sm, 2*sm*tr/(sm+tr), matched_length[i] / exact[i], matched_length[i], rc, pr, sm_100tr, rc_100tr, pr_100tr, al))
if __name__ == '__main__':
    main()
